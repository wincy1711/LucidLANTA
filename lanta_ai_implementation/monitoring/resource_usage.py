#!/usr/bin/env python3
"""
Resource usage monitoring script for LANTA cluster
Tracks CPU, memory, disk, and network usage during job execution
"""

import os
import time
import json
import argparse
import threading
import subprocess
from datetime import datetime
from pathlib import Path

import psutil
import numpy as np

class ResourceMonitor:
    """Resource usage monitoring class"""
    
    def __init__(self, output_dir="resource_monitor", interval=5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.interval = interval
        self.running = False
        self.data = []
        self.start_time = None
        
        # Process monitoring
        self.target_processes = []
        self.monitor_children = True
    
    def set_target_processes(self, process_names):
        """Set target processes to monitor"""
        self.target_processes = process_names
    
    def find_processes_by_name(self, name):
        """Find processes by name"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if name.lower() in proc.info['name'].lower():
                    processes.append(proc)
                elif proc.info['cmdline'] and any(name.lower() in cmd.lower() for cmd in proc.info['cmdline']):
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def get_process_metrics(self, process):
        """Get metrics for a specific process"""
        try:
            with process.oneshot():
                # Basic info
                pid = process.pid
                name = process.name()
                status = process.status()
                
                # CPU and memory
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # I/O
                io_counters = process.io_counters()
                
                # Threads
                num_threads = process.num_threads()
                
                # Open files
                try:
                    open_files = len(process.open_files())
                except:
                    open_files = 0
                
                return {
                    "pid": pid,
                    "name": name,
                    "status": status,
                    "cpu_percent": cpu_percent,
                    "memory_rss_mb": memory_info.rss / (1024 * 1024),
                    "memory_vms_mb": memory_info.vms / (1024 * 1024),
                    "memory_percent": memory_percent,
                    "num_threads": num_threads,
                    "open_files": open_files,
                    "io_read_bytes": io_counters.read_bytes,
                    "io_write_bytes": io_counters.write_bytes,
                    "io_read_count": io_counters.read_count,
                    "io_write_count": io_counters.write_count
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def get_system_metrics(self):
        """Get system-wide metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_stats = psutil.cpu_stats()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Load average (Unix systems)
        try:
            load_avg = os.getloadavg()
        except:
            load_avg = None
        
        return {
            "timestamp": timestamp,
            "elapsed_time": timestamp - self.start_time if self.start_time else 0,
            "cpu": {
                "percent_per_core": cpu_percent,
                "percent_avg": np.mean(cpu_percent),
                "freq_mhz": cpu_freq.current if cpu_freq else None,
                "freq_max_mhz": cpu_freq.max if cpu_freq else None,
                "ctx_switches": cpu_stats.ctx_switches,
                "interrupts": cpu_stats.interrupts,
                "soft_interrupts": cpu_stats.soft_interrupts,
                "syscalls": cpu_stats.syscalls
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "free_gb": memory.free / (1024**3),
                "percent": memory.percent,
                "active_gb": memory.active / (1024**3) if hasattr(memory, 'active') else None,
                "inactive_gb": memory.inactive / (1024**3) if hasattr(memory, 'inactive') else None
            },
            "swap": {
                "total_gb": swap.total / (1024**3),
                "used_gb": swap.used / (1024**3),
                "free_gb": swap.free / (1024**3),
                "percent": swap.percent
            },
            "disk": {
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "percent": disk_usage.percent,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0
            },
            "network": {
                "bytes_sent_gb": network_io.bytes_sent / (1024**3),
                "bytes_recv_gb": network_io.bytes_recv / (1024**3),
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
                "errin": network_io.errin,
                "errout": network_io.errout,
                "dropin": network_io.dropin,
                "dropout": network_io.dropout
            },
            "load_average": load_avg
        }
    
    def get_process_tree_metrics(self):
        """Get metrics for target processes and their children"""
        process_metrics = {}
        
        for proc_name in self.target_processes:
            processes = self.find_processes_by_name(proc_name)
            
            for process in processes:
                # Get metrics for main process
                metrics = self.get_process_metrics(process)
                if metrics:
                    process_metrics[f"{proc_name}_{process.pid}"] = metrics
                
                # Get metrics for children if requested
                if self.monitor_children:
                    try:
                        children = process.children(recursive=True)
                        for child in children:
                            child_metrics = self.get_process_metrics(child)
                            if child_metrics:
                                process_metrics[f"{proc_name}_child_{child.pid}"] = child_metrics
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        
        return process_metrics
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.get_system_metrics()
                
                # Collect process metrics
                process_metrics = self.get_process_tree_metrics()
                
                # Combine metrics
                entry = {
                    "system": system_metrics,
                    "processes": process_metrics
                }
                
                self.data.append(entry)
                
                # Sleep for interval
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
    
    def start_monitoring(self, duration=None, processes=None):
        """Start monitoring"""
        if processes:
            self.set_target_processes(processes)
        
        print(f"Starting resource monitoring...")
        if self.target_processes:
            print(f"Target processes: {', '.join(self.target_processes)}")
        print(f"Monitoring interval: {self.interval} seconds")
        
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
            filename = f"resource_monitor_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data for saving
        save_data = {
            "metadata": {
                "start_time": self.start_time,
                "end_time": time.time(),
                "duration": time.time() - self.start_time if self.start_time else 0,
                "interval": self.interval,
                "total_samples": len(self.data),
                "target_processes": self.target_processes,
                "monitor_children": self.monitor_children
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
            "duration": self.data[-1]["system"]["elapsed_time"] if self.data else 0,
            "total_samples": len(self.data)
        }
        
        # System metrics summary
        system_data = [d["system"] for d in self.data]
        
        summary["system"] = {
            "avg_cpu_percent": np.mean([d["cpu"]["percent_avg"] for d in system_data]),
            "max_cpu_percent": np.max([d["cpu"]["percent_avg"] for d in system_data]),
            "avg_memory_percent": np.mean([d["memory"]["percent"] for d in system_data]),
            "max_memory_percent": np.max([d["memory"]["percent"] for d in system_data]),
            "avg_disk_percent": np.mean([d["disk"]["percent"] for d in system_data]),
            "max_disk_percent": np.max([d["disk"]["percent"] for d in system_data]),
            "total_network_sent_gb": system_data[-1]["network"]["bytes_sent_gb"] - system_data[0]["network"]["bytes_sent_gb"],
            "total_network_recv_gb": system_data[-1]["network"]["bytes_recv_gb"] - system_data[0]["network"]["bytes_recv_gb"]
        }
        
        # Process metrics summary
        if self.data and "processes" in self.data[0] and self.data[0]["processes"]:
            summary["processes"] = {}
            
            # Get all process names
            all_processes = set()
            for entry in self.data:
                if "processes" in entry:
                    all_processes.update(entry["processes"].keys())
            
            for proc_name in all_processes:
                proc_data = []
                for entry in self.data:
                    if "processes" in entry and proc_name in entry["processes"]:
                        proc_data.append(entry["processes"][proc_name])
                
                if proc_data:
                    summary["processes"][proc_name] = {
                        "avg_cpu_percent": np.mean([d["cpu_percent"] for d in proc_data if "cpu_percent" in d]),
                        "max_cpu_percent": np.max([d["cpu_percent"] for d in proc_data if "cpu_percent" in d]),
                        "avg_memory_rss_mb": np.mean([d["memory_rss_mb"] for d in proc_data if "memory_rss_mb" in d]),
                        "max_memory_rss_mb": np.max([d["memory_rss_mb"] for d in proc_data if "memory_rss_mb" in d]),
                        "avg_memory_percent": np.mean([d["memory_percent"] for d in proc_data if "memory_percent" in d]),
                        "max_memory_percent": np.max([d["memory_percent"] for d in proc_data if "memory_percent" in d]),
                        "avg_num_threads": np.mean([d["num_threads"] for d in proc_data if "num_threads" in d]),
                        "max_num_threads": np.max([d["num_threads"] for d in proc_data if "num_threads" in d])
                    }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("RESOURCE MONITORING SUMMARY")
        print("="*60)
        
        print(f"Duration: {summary.get('duration', 0):.2f} seconds")
        print(f"Total Samples: {summary.get('total_samples', 0)}")
        
        # System summary
        if "system" in summary:
            sys_summary = summary["system"]
            print(f"\nSystem Metrics:")
            print(f"  CPU Usage: {sys_summary['avg_cpu_percent']:.1f}% avg, {sys_summary['max_cpu_percent']:.1f}% max")
            print(f"  Memory Usage: {sys_summary['avg_memory_percent']:.1f}% avg, {sys_summary['max_memory_percent']:.1f}% max")
            print(f"  Disk Usage: {sys_summary['avg_disk_percent']:.1f}% avg, {sys_summary['max_disk_percent']:.1f}% max")
            print(f"  Network: {sys_summary['total_network_sent_gb']:.2f} GB sent, {sys_summary['total_network_recv_gb']:.2f} GB received")
        
        # Process summary
        if "processes" in summary:
            print(f"\nProcess Metrics:")
            for proc_name, proc_summary in summary["processes"].items():
                print(f"  {proc_name}:")
                if "avg_cpu_percent" in proc_summary:
                    print(f"    CPU: {proc_summary['avg_cpu_percent']:.1f}% avg, {proc_summary['max_cpu_percent']:.1f}% max")
                if "avg_memory_rss_mb" in proc_summary:
                    print(f"    Memory RSS: {proc_summary['avg_memory_rss_mb']:.0f} MB avg, {proc_summary['max_memory_rss_mb']:.0f} MB max")
                if "avg_memory_percent" in proc_summary:
                    print(f"    Memory %: {proc_summary['avg_memory_percent']:.2f}% avg, {proc_summary['max_memory_percent']:.2f}% max")
                if "avg_num_threads" in proc_summary:
                    print(f"    Threads: {proc_summary['avg_num_threads']:.0f} avg, {proc_summary['max_num_threads']:.0f} max")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Resource Usage Monitor')
    parser.add_argument('--output-dir', type=str, default='resource_monitor', help='Output directory')
    parser.add_argument('--interval', type=float, default=5.0, help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds')
    parser.add_argument('--processes', nargs='+', help='Process names to monitor')
    parser.add_argument('--no-children', action='store_true', help='Do not monitor child processes')
    parser.add_argument('--save', action='store_true', help='Save monitoring data')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ResourceMonitor(output_dir=args.output_dir, interval=args.interval)
    
    # Configure child process monitoring
    if args.no_children:
        monitor.monitor_children = False
    
    # Start monitoring
    monitor.start_monitoring(duration=args.duration, processes=args.processes)
    
    # Save data if requested
    if args.save:
        monitor.save_data()
    
    # Print summary
    monitor.print_summary()

if __name__ == '__main__':
    main()