#!/usr/bin/env python3
"""
GPU profiling script for LANTA cluster
Monitors GPU usage, memory, and performance metrics
"""

import os
import time
import json
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import psutil
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py")

class GPUProfiler:
    """GPU profiling and monitoring class"""
    
    def __init__(self, output_dir="gpu_profile", interval=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.interval = interval
        self.running = False
        self.data = []
        self.start_time = None
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"Found {self.gpu_count} GPU(s)")
            except pynvml.NVMLError as e:
                print(f"Failed to initialize NVML: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def get_gpu_info(self):
        """Get basic GPU information"""
        if not PYNVML_AVAILABLE or self.gpu_count == 0:
            return {}
        
        gpu_info = {}
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info[f"gpu_{i}"] = {
                    "name": name,
                    "total_memory_mb": memory_info.total // (1024 * 1024),
                    "driver_version": pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                    "cuda_version": pynvml.nvmlSystemGetCudaDriverVersion()
                }
            except pynvml.NVMLError as e:
                print(f"Error getting GPU {i} info: {e}")
        
        return gpu_info
    
    def collect_gpu_metrics(self):
        """Collect GPU metrics"""
        if not PYNVML_AVAILABLE or self.gpu_count == 0:
            return {}
        
        metrics = {}
        timestamp = time.time()
        
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = 0
                    memory_clock = 0
                
                metrics[f"gpu_{i}"] = {
                    "timestamp": timestamp,
                    "memory_used_mb": memory_info.used // (1024 * 1024),
                    "memory_free_mb": memory_info.free // (1024 * 1024),
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "temperature_c": temperature,
                    "power_watts": power,
                    "graphics_clock_mhz": graphics_clock,
                    "memory_clock_mhz": memory_clock
                }
            except pynvml.NVMLError as e:
                print(f"Error collecting GPU {i} metrics: {e}")
        
        return metrics
    
    def collect_system_metrics(self):
        """Collect system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        return {
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "network_sent_gb": net_io.bytes_sent / (1024**3),
            "network_recv_gb": net_io.bytes_recv / (1024**3)
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                timestamp = time.time()
                
                # Collect metrics
                gpu_metrics = self.collect_gpu_metrics()
                system_metrics = self.collect_system_metrics()
                
                # Combine metrics
                entry = {
                    "timestamp": timestamp,
                    "elapsed_time": timestamp - self.start_time if self.start_time else 0,
                    "system": system_metrics,
                    "gpus": gpu_metrics
                }
                
                self.data.append(entry)
                
                # Sleep for interval
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
    
    def start_monitoring(self, duration=None):
        """Start monitoring"""
        print("Starting GPU profiling...")
        self.running = True
        self.start_time = time.time()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        if duration:
            print(f"Monitoring for {duration} seconds...")
            time.sleep(duration)
            self.stop_monitoring()
        else:
            print("Monitoring indefinitely. Press Ctrl+C to stop.")
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                self.stop_monitoring()
        
        monitor_thread.join()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
    
    def save_data(self, filename=None):
        """Save collected data"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_profile_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data for saving
        save_data = {
            "metadata": {
                "start_time": self.start_time,
                "end_time": time.time(),
                "duration": time.time() - self.start_time if self.start_time else 0,
                "gpu_count": self.gpu_count,
                "gpu_info": self.get_gpu_info(),
                "interval": self.interval,
                "total_samples": len(self.data)
            },
            "data": self.data
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Data saved to {filepath}")
        return filepath
    
    def generate_summary(self):
        """Generate summary statistics"""
        if not self.data:
            return {}
        
        summary = {
            "duration": self.data[-1]["elapsed_time"] if self.data else 0,
            "total_samples": len(self.data)
        }
        
        # System metrics summary
        if self.data and "system" in self.data[0]:
            system_data = [d["system"] for d in self.data]
            summary["system"] = {
                "avg_cpu_percent": np.mean([d["cpu_percent"] for d in system_data]),
                "max_cpu_percent": np.max([d["cpu_percent"] for d in system_data]),
                "avg_memory_percent": np.mean([d["memory_percent"] for d in system_data]),
                "max_memory_percent": np.max([d["memory_percent"] for d in system_data])
            }
        
        # GPU metrics summary
        if self.gpu_count > 0 and self.data and "gpus" in self.data[0]:
            summary["gpus"] = {}
            
            for i in range(self.gpu_count):
                gpu_key = f"gpu_{i}"
                gpu_data = [d["gpus"][gpu_key] for d in self.data if gpu_key in d["gpus"]]
                
                if gpu_data:
                    summary["gpus"][gpu_key] = {
                        "avg_gpu_utilization": np.mean([d["gpu_utilization"] for d in gpu_data]),
                        "max_gpu_utilization": np.max([d["gpu_utilization"] for d in gpu_data]),
                        "avg_memory_utilization": np.mean([d["memory_utilization"] for d in gpu_data]),
                        "max_memory_utilization": np.max([d["memory_utilization"] for d in gpu_data]),
                        "avg_temperature": np.mean([d["temperature_c"] for d in gpu_data]),
                        "max_temperature": np.max([d["temperature_c"] for d in gpu_data]),
                        "avg_power": np.mean([d["power_watts"] for d in gpu_data]),
                        "max_power": np.max([d["power_watts"] for d in gpu_data])
                    }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("GPU PROFILING SUMMARY")
        print("="*60)
        
        print(f"Duration: {summary.get('duration', 0):.2f} seconds")
        print(f"Total Samples: {summary.get('total_samples', 0)}")
        
        # System summary
        if "system" in summary:
            sys_summary = summary["system"]
            print(f"\nSystem Metrics:")
            print(f"  CPU Usage: {sys_summary['avg_cpu_percent']:.1f}% avg, {sys_summary['max_cpu_percent']:.1f}% max")
            print(f"  Memory Usage: {sys_summary['avg_memory_percent']:.1f}% avg, {sys_summary['max_memory_percent']:.1f}% max")
        
        # GPU summary
        if "gpus" in summary:
            print(f"\nGPU Metrics:")
            for gpu_id, gpu_summary in summary["gpus"].items():
                print(f"  {gpu_id.upper()}:")
                print(f"    GPU Utilization: {gpu_summary['avg_gpu_utilization']:.1f}% avg, {gpu_summary['max_gpu_utilization']:.1f}% max")
                print(f"    Memory Utilization: {gpu_summary['avg_memory_utilization']:.1f}% avg, {gpu_summary['max_memory_utilization']:.1f}% max")
                print(f"    Temperature: {gpu_summary['avg_temperature']:.1f}°C avg, {gpu_summary['max_temperature']:.1f}°C max")
                print(f"    Power: {gpu_summary['avg_power']:.1f}W avg, {gpu_summary['max_power']:.1f}W max")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='GPU Profiling Tool')
    parser.add_argument('--output-dir', type=str, default='gpu_profile', help='Output directory')
    parser.add_argument('--interval', type=float, default=1.0, help='Sampling interval in seconds')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds')
    parser.add_argument('--save', action='store_true', help='Save profiling data')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = GPUProfiler(output_dir=args.output_dir, interval=args.interval)
    
    # Check GPU availability
    if profiler.gpu_count == 0:
        print("No GPUs detected or pynvml not available")
        return
    
    # Start monitoring
    profiler.start_monitoring(duration=args.duration)
    
    # Save data if requested
    if args.save:
        profiler.save_data()
    
    # Print summary
    profiler.print_summary()

if __name__ == '__main__':
    main()