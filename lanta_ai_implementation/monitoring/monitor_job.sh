#!/bin/bash
# Job monitoring script for LANTA cluster
# Provides real-time monitoring of job status and resource usage

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}%s${NC}\n" "$2"
}

# Function to check if job exists
check_job_exists() {
    local job_id=$1
    if squeue -j $job_id 2>/dev/null | grep -q $job_id; then
        return 0
    else
        return 1
    fi
}

# Function to get job information
get_job_info() {
    local job_id=$1
    scontrol show job $job_id 2>/dev/null | grep -E "JobId|JobState|Partition|NodeList|TimeLimit|StartTime"
}

# Function to monitor GPU usage
monitor_gpu_usage() {
    local node=$1
    echo "GPU Usage on $node:"
    ssh $node "nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "Unable to connect to $node"
}

# Function to monitor CPU and memory usage
monitor_system_usage() {
    local node=$1
    echo "System Usage on $node:"
    ssh $node "top -bn1 | grep 'Cpu(s)' && free -h | grep 'Mem:'" 2>/dev/null || echo "Unable to connect to $node"
}

# Function to show job logs
show_job_logs() {
    local job_id=$1
    local log_file=$(find . -name "slurm_${job_id}.out" -o -name "slurm_${job_id}_*.out" | head -1)
    
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "Recent log entries:"
        tail -20 "$log_file"
    else
        echo "No log file found for job $job_id"
    fi
}

# Function to monitor job in real-time
monitor_job_realtime() {
    local job_id=$1
    local interval=${2:-10}  # Default 10 seconds
    
    print_color $BLUE "Starting real-time monitoring of job $job_id (interval: ${interval}s)"
    print_color $YELLOW "Press Ctrl+C to stop monitoring"
    
    while true; do
        clear
        echo "========================================="
        print_color $GREEN "Job Monitor - $(date)"
        echo "========================================="
        
        # Check if job still exists
        if ! check_job_exists $job_id; then
            print_color $RED "Job $job_id is no longer running"
            echo "Final job status:"
            sacct -j $job_id --format=JobID,JobName,State,ExitCode,MaxRSS,MaxVMSize,Elapsed
            break
        fi
        
        # Get job information
        echo "Job Information:"
        get_job_info $job_id
        echo ""
        
        # Get job status from squeue
        echo "Current Status:"
        squeue -j $job_id -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
        echo ""
        
        # Get node information
        local nodes=$(squeue -j $job_id -h -o "%N")
        if [ -n "$nodes" ]; then
            echo "Node Information:"
            for node in $nodes; do
                if [ "$node" != "N/A" ]; then
                    echo "Node: $node"
                    monitor_gpu_usage $node
                    echo ""
                fi
            done
        fi
        
        # Show recent logs
        echo "Recent Logs:"
        show_job_logs $job_id
        echo ""
        
        echo "========================================="
        print_color $YELLOW "Next update in ${interval} seconds..."
        sleep $interval
    done
}

# Function to show job summary
show_job_summary() {
    local job_id=$1
    
    echo "========================================="
    print_color $GREEN "Job Summary for $job_id"
    echo "========================================="
    
    # Job information
    echo "Job Details:"
    sacct -j $job_id --format=JobID,JobName,Partition,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize
    echo ""
    
    # Resource usage
    echo "Resource Usage:"
    seff $job_id 2>/dev/null || echo "seff not available for this job"
    echo ""
    
    # Node information
    echo "Node Information:"
    scontrol show job $job_id 2>/dev/null | grep NodeList || echo "Node information not available"
}

# Function to monitor multiple jobs
monitor_multiple_jobs() {
    local job_ids=($@)
    
    echo "========================================="
    print_color $GREEN "Monitoring Multiple Jobs"
    echo "========================================="
    
    for job_id in "${job_ids[@]}"; do
        echo "Job $job_id:"
        if check_job_exists $job_id; then
            squeue -j $job_id -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
        else
            print_color $RED "Job $job_id is not running"
        fi
        echo ""
    done
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <job_id> [interval]"
        echo "       $0 summary <job_id>"
        echo "       $0 multi <job_id1> <job_id2> ..."
        echo ""
        echo "Commands:"
        echo "  <job_id>        Monitor single job in real-time"
        echo "  summary <job_id> Show job summary"
        echo "  multi <job_ids> Monitor multiple jobs"
        echo ""
        echo "Examples:"
        echo "  $0 12345 5      # Monitor job 12345 every 5 seconds"
        echo "  $0 summary 12345 # Show summary for job 12345"
        echo "  $0 multi 12345 12346 12347  # Monitor multiple jobs"
        exit 1
    fi
    
    case $1 in
        summary)
            if [ -z "$2" ]; then
                echo "Error: Job ID required for summary"
                exit 1
            fi
            show_job_summary $2
            ;;
        multi)
            if [ $# -lt 2 ]; then
                echo "Error: At least one job ID required for multi monitoring"
                exit 1
            fi
            monitor_multiple_jobs "${@:2}"
            ;;
        *)
            job_id=$1
            interval=${2:-10}
            
            # Validate job ID
            if ! [[ $job_id =~ ^[0-9]+$ ]]; then
                echo "Error: Invalid job ID format"
                exit 1
            fi
            
            # Check if job exists
            if ! check_job_exists $job_id; then
                echo "Job $job_id not found or not running"
                echo "Checking job history..."
                sacct -j $job_id --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
                exit 1
            fi
            
            # Start monitoring
            monitor_job_realtime $job_id $interval
            ;;
    esac
}

# Run main function
main "$@"