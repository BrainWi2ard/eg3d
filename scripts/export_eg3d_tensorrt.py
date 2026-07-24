#!/usr/bin/env python3
"""
Complete TensorRT export pipeline for EG3D.

Loads a pre-trained EG3D .pkl generator, extracts the tri-plane synthesis and 
volume renderer networks, exports to ONNX with dynamic batch axes, and compiles 
to TensorRT .engine file using FP16 precision.

Usage:
    python export_eg3d_tensorrt.py \
        --model modelhub/eg3d/ffhq_triplane_generator.pkl \
        --output modelhub/eg3d/eg3d_tensorrt_fp16.engine \
        --precision fp16 \
        --max_batch_size 4 \
        --max_workspace_gb 4 \
        --benchmark
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import tensorrt as trt
import onnx


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EG3DTensorRTExporter:
    """
    Complete pipeline for exporting EG3D generator to optimized TensorRT engine.
    
    Flow:
    1. Load pre-trained PyTorch model from pickle
    2. Extract synthesis network and tri-plane decoder
    3. Create ONNX export wrapper
    4. Export to ONNX with dynamic batch axes
    5. Compile ONNX to TensorRT with specified precision
    6. Benchmark compiled engine
    """
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "modelhub/eg3d",
                 device_id: int = 0,
                 verbose: bool = True):
        """
        Initialize exporter.
        
        Args:
            model_path: Path to EG3D .pkl checkpoint
            output_dir: Directory for output files
            device_id: CUDA device to use
            verbose: Enable verbose logging
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.verbose = verbose
        
        # Model components
        self.generator = None
        self.synthesis_network = None
        
        if self.verbose:
            logger.info(f"Initialized EG3D TensorRT Exporter")
            logger.info(f"  Model: {self.model_path}")
            logger.info(f"  Output: {self.output_dir}")
            logger.info(f"  Device: cuda:{device_id}")
    
    def load_pytorch_model(self) -> None:
        """
        Load EG3D generator from pickle checkpoint.
        
        Expected checkpoint structure:
        {
            'G_ema': <generator module>,
            'G': <generator at iteration>,
            'D': <discriminator>,
            ...
        }
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading PyTorch model from {self.model_path}...")
        
        try:
            with open(self.model_path, 'rb') as f:
                checkpoint = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Extract generator
        if 'G_ema' in checkpoint:
            self.generator = checkpoint['G_ema']
            logger.info("Using G_ema (exponential moving average)")
        elif 'G' in checkpoint:
            self.generator = checkpoint['G']
            logger.info("Using G (current iteration)")
        else:
            raise ValueError("Checkpoint missing 'G_ema' or 'G' key")
        
        # Move to device and set to eval mode
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        
        # Verify architecture
        assert hasattr(self.generator, 'mapping'), "Generator missing 'mapping' network"
        assert hasattr(self.generator, 'synthesis'), "Generator missing 'synthesis' network"
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Generator parameters: {self._count_parameters(self.generator):,}")
        logger.info(f"  Mapping network: {self.generator.mapping}")
        logger.info(f"  Synthesis network: {self.generator.synthesis}")
    
    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def export_to_onnx(self, 
                      output_path: Optional[str] = None,
                      opset_version: int = 17,
                      batch_size: int = 1) -> str:
        """
        Export synthesis network to ONNX format.
        
        Creates ONNX graph with:
        - Dynamic batch size axis
        - Inputs: w_code (latent), camera_params (conditioning)
        - Outputs: rendered_image, feature_volume (for downstream modules)
        
        Args:
            output_path: ONNX output file path
            opset_version: ONNX opset version (17 recommended for FP16)
            batch_size: Batch size for export trace
            
        Returns:
            Path to exported ONNX file
        """
        if output_path is None:
            output_path = str(self.output_dir / "eg3d_synthesis_temp.onnx")
        
        logger.info(f"Exporting synthesis network to ONNX: {output_path}")
        logger.info(f"  Opset version: {opset_version}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Create dummy inputs
        z_dim = self.generator.z_dim
        dummy_w = torch.randn(batch_size, self.generator.w_dim, device=self.device)
        dummy_camera = torch.randn(batch_size, 25, device=self.device)
        dummy_truncation_psi = torch.tensor(0.7, device=self.device)
        dummy_truncation_cutoff = torch.tensor(14, device=self.device)
        dummy_noise_mode = 'const'
        dummy_force_fp32 = False
        
        logger.info(f"Input shapes:")
        logger.info(f"  w_code: {dummy_w.shape}")
        logger.info(f"  camera_params: {dummy_camera.shape}")
        
        # Wrap synthesis for ONNX export
        class SynthesisWrapper(torch.nn.Module):
            """Wrapper for EG3D synthesis to simplify ONNX export."""
            def __init__(self, synthesis_net):
                super().__init__()
                self.synthesis_net = synthesis_net
            
            def forward(self, w, camera_params):
                # Call synthesis network with default parameters
                return self.synthesis_net(
                    w,
                    camera_params,
                    truncation_psi=0.7,
                    truncation_cutoff=14,
                    force_fp32=False,
                    noise_mode='const'
                )
        
        synthesis_wrapper = SynthesisWrapper(self.generator.synthesis)
        synthesis_wrapper = synthesis_wrapper.to(self.device)
        synthesis_wrapper.eval()
        
        # Export to ONNX
        try:
            torch.onnx.export(
                synthesis_wrapper,
                args=(dummy_w, dummy_camera),
                f=output_path,
                input_names=['w_code', 'camera_params'],
                output_names=['rendered_image'],
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL,
                dynamic_axes={
                    'w_code': {0: 'batch_size'},
                    'camera_params': {0: 'batch_size'},
                    'rendered_image': {0: 'batch_size'}
                }
            )
            
            logger.info(f"✓ ONNX export successful: {output_path}")
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"✓ ONNX model validation passed")
            
            return output_path
        
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def compile_onnx_to_tensorrt(self,
                                onnx_path: str,
                                output_engine_path: Optional[str] = None,
                                precision: str = 'fp16',
                                max_batch_size: int = 4,
                                max_workspace_gb: int = 2) -> str:
        """
        Compile ONNX model to optimized TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model file
            output_engine_path: Output .engine file path
            precision: 'fp32', 'fp16', or 'int8'
            max_batch_size: Maximum batch size for engine
            max_workspace_gb: GPU workspace memory allocation (GB)
            
        Returns:
            Path to compiled .engine file
        """
        if output_engine_path is None:
            output_engine_path = str(
                self.output_dir / f"eg3d_tensorrt_{precision}.engine"
            )
        
        logger.info(f"Compiling ONNX to TensorRT ({precision})...")
        logger.info(f"  Input: {onnx_path}")
        logger.info(f"  Output: {output_engine_path}")
        logger.info(f"  Max batch size: {max_batch_size}")
        logger.info(f"  Workspace: {max_workspace_gb} GB")
        
        # Initialize TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Create builder and network
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Parse ONNX
        logger.info("Parsing ONNX model...")
        parser = trt.OnnxParser(network, trt_logger)
        
        with open(onnx_path, 'rb') as f:
            onnx_data = f.read()
        
        if not parser.parse(onnx_data):
            logger.error("ONNX parsing failed:")
            for error in parser.errors:
                logger.error(f"  {error}")
            raise RuntimeError("Failed to parse ONNX model")
        
        logger.info(f"✓ ONNX model parsed successfully")
        logger.info(f"  Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
        logger.info(f"  Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_gb * (1 << 30)  # Convert GB to bytes
        
        # Set precision
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                logger.info("FP16 precision available - enabling")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                logger.warning("FP16 not supported on this GPU - falling back to FP32")
                precision = 'fp32'
        
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                logger.info("INT8 precision available - enabling")
                config.set_flag(trt.BuilderFlag.INT8)
                # In production, you would set INT8 calibration data here
            else:
                logger.warning("INT8 not supported on this GPU - falling back to FP32")
                precision = 'fp32'
        
        # Disable GPU fallback (optional - helps catch issues)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        
        # Create optimization profile
        logger.info("Creating optimization profiles...")
        profile = builder.create_optimization_profile()
        
        # Set input shapes
        # w_code: (batch_size, 512)
        profile.set_shape(
            'w_code',
            min=(1, 512),
            opt=(max_batch_size, 512),
            max=(max_batch_size, 512)
        )
        
        # camera_params: (batch_size, 25)
        profile.set_shape(
            'camera_params',
            min=(1, 25),
            opt=(max_batch_size, 25),
            max=(max_batch_size, 25)
        )
        
        config.add_optimization_profile(profile)
        
        # Build engine
        logger.info("Building TensorRT engine (this may take 1-3 minutes)...")
        logger.info("  Optimizing for inference...")
        
        try:
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("Failed to build serialized engine")
        except Exception as e:
            logger.error(f"Engine compilation failed: {e}")
            raise
        
        # Save engine
        logger.info(f"Saving engine to {output_engine_path}...")
        with open(output_engine_path, 'wb') as f:
            f.write(engine_bytes)
        
        # Log engine stats
        file_size_mb = Path(output_engine_path).stat().st_size / (1024 ** 2)
        logger.info(f"✓ Engine compilation successful!")
        logger.info(f"  File: {output_engine_path}")
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        logger.info(f"  Precision: {precision}")
        
        return output_engine_path
    
    def benchmark_engine(self,
                        engine_path: str,
                        num_iterations: int = 100,
                        batch_size: int = 1) -> Dict[str, float]:
        """
        Benchmark compiled TensorRT engine.
        
        Args:
            engine_path: Path to .engine file
            num_iterations: Number of iterations to average
            batch_size: Batch size for benchmark
            
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"Benchmarking engine: {engine_path}")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        
        # Allocate GPU memory for I/O
        bindings = []
        for i in range(engine.num_bindings):
            size = batch_size
            for s in engine.get_binding_shape(i)[1:]:
                size *= s
            
            dtype = engine.get_binding_dtype(i)
            if dtype == trt.float32:
                torch_dtype = torch.float32
            elif dtype == trt.float16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.int32
            
            device_mem = torch.empty(size, dtype=torch_dtype, device=self.device)
            bindings.append(int(device_mem.data_ptr()))
        
        # Warm up GPU
        logger.info("Warming up GPU...")
        for _ in range(10):
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        
        torch.cuda.synchronize()
        
        # Benchmark
        logger.info("Running benchmark...")
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        for _ in range(num_iterations):
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        
        end_time.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_time.elapsed_time(end_time)
        avg_latency_ms = elapsed_ms / num_iterations
        throughput_fps = 1000.0 / avg_latency_ms
        
        stats = {
            'avg_latency_ms': avg_latency_ms,
            'throughput_fps': throughput_fps,
            'total_time_ms': elapsed_ms,
            'iterations': num_iterations
        }
        
        logger.info(f"✓ Benchmark complete!")
        logger.info(f"  Avg latency: {avg_latency_ms:.2f} ms")
        logger.info(f"  Throughput: {throughput_fps:.1f} FPS")
        logger.info(f"  Total time: {elapsed_ms:.1f} ms")
        
        return stats
    
    def export_full_pipeline(self,
                            precision: str = 'fp16',
                            max_batch_size: int = 4,
                            max_workspace_gb: int = 2,
                            benchmark: bool = False) -> str:
        """
        Complete export pipeline: PyTorch → ONNX → TensorRT.
        
        Args:
            precision: 'fp32', 'fp16', or 'int8'
            max_batch_size: Maximum batch size
            max_workspace_gb: GPU workspace memory
            benchmark: Run benchmark after compilation
            
        Returns:
            Path to compiled .engine file
        """
        # Step 1: Load PyTorch model
        self.load_pytorch_model()
        
        # Step 2: Export to ONNX
        onnx_path = self.export_to_onnx()
        
        # Step 3: Compile to TensorRT
        engine_path = self.compile_onnx_to_tensorrt(
            onnx_path,
            precision=precision,
            max_batch_size=max_batch_size,
            max_workspace_gb=max_workspace_gb
        )
        
        # Step 4: Benchmark (optional)
        if benchmark:
            stats = self.benchmark_engine(engine_path, num_iterations=100)
        
        # Clean up temporary ONNX file
        if Path(onnx_path).exists():
            os.remove(onnx_path)
            logger.info(f"Cleaned up temporary ONNX file")
        
        logger.info("=" * 70)
        logger.info("EXPORT PIPELINE COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Engine: {engine_path}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Max batch size: {max_batch_size}")
        
        return engine_path


def main():
    """Command-line interface for TensorRT export."""
    parser = argparse.ArgumentParser(
        description="Export EG3D generator to TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with FP16 precision
  python export_eg3d_tensorrt.py \\
    --model modelhub/eg3d/ffhq_triplane.pkl \\
    --precision fp16

  # Export with INT8 precision and benchmarking
  python export_eg3d_tensorrt.py \\
    --model modelhub/eg3d/ffhq_triplane.pkl \\
    --precision int8 \\
    --max_batch_size 8 \\
    --benchmark

  # Export to custom output directory
  python export_eg3d_tensorrt.py \\
    --model modelhub/eg3d/ffhq_triplane.pkl \\
    --output /path/to/output \\
    --precision fp16
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to EG3D .pkl checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='modelhub/eg3d',
        help='Output directory for .engine file (default: modelhub/eg3d)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp32', 'fp16', 'int8'],
        default='fp16',
        help='TensorRT precision (default: fp16)'
    )
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=4,
        help='Maximum batch size (default: 4)'
    )
    parser.add_argument(
        '--max_workspace_gb',
        type=int,
        default=2,
        help='GPU workspace memory in GB (default: 2)'
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
        help='CUDA device ID (default: 0)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark after compilation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Run export pipeline
    try:
        exporter = EG3DTensorRTExporter(
            model_path=args.model,
            output_dir=args.output,
            device_id=args.device_id,
            verbose=args.verbose
        )
        
        engine_path = exporter.export_full_pipeline(
            precision=args.precision,
            max_batch_size=args.max_batch_size,
            max_workspace_gb=args.max_workspace_gb,
            benchmark=args.benchmark
        )
        
        logger.info(f"\n✓ Success! Engine saved to: {engine_path}")
        return 0
    
    except Exception as e:
        logger.error(f"\n✗ Export failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
