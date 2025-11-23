#!/bin/bash
# Cleanup utilities for LANTA cluster
# Safely cleans up temporary files and old job data

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

# Function to check if user wants to proceed
confirm_action() {
    local message=$1
    print_color $YELLOW "$message"
    read -p "Do you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_color $RED "Action cancelled"
        return 1
    fi
    return 0
}

# Function to safely remove files
safe_remove() {
    local path=$1
    local description=$2
    
    if [ -e "$path" ]; then
        local size=$(du -sh "$path" 2>/dev/null | cut -f1 || echo "unknown")
        print_color $BLUE "Removing $description: $path (size: $size)"
        
        if [ -d "$path" ]; then
            rm -rf "$path"
        else
            rm -f "$path"
        fi
        
        print_color $GREEN "Removed: $path"
    else
        print_color $YELLOW "Path does not exist: $path"
    fi
}

# Function to cleanup temporary files
cleanup_temp_files() {
    print_color $BLUE "Cleaning up temporary files..."
    
    # Common temporary file locations
    local temp_dirs=(
        "/tmp"
        "/var/tmp"
        "$TMPDIR"
        "/scratch/$USER/tmp"
    )
    
    # Common temporary file patterns
    local temp_patterns=(
        "*.tmp"
        "*.temp"
        "temp_*"
        "tmp_*"
        ".*.swp"
        ".*.swo"
        "*~"
        "core"
        "*.log"
        "*.out"
        "*.err"
    )
    
    # Cleanup in temp directories
    for temp_dir in "${temp_dirs[@]}"; do
        if [ -d "$temp_dir" ]; then
            print_color $YELLOW "Cleaning temp directory: $temp_dir"
            
            for pattern in "${temp_patterns[@]}"; do
                find "$temp_dir" -name "$pattern" -type f -mtime +1 -delete 2>/dev/null || true
            done
        fi
    done
    
    # Cleanup in current directory and subdirectories
    print_color $YELLOW "Cleaning current directory..."
    for pattern in "${temp_patterns[@]}"; do
        find . -name "$pattern" -type f -mtime +7 -delete 2>/dev/null || true
    done
    
    print_color $GREEN "Temporary files cleanup completed"
}

# Function to cleanup old job data
cleanup_job_data() {
    local retention_days=${1:-30}
    
    print_color $BLUE "Cleaning up old job data (older than $retention_days days)..."
    
    # Common job data locations
    local job_dirs=(
        "/scratch/$USER"
        "$HOME/slurm_logs"
        "$HOME/job_outputs"
        "$HOME/experiments"
    )
    
    for job_dir in "${job_dirs[@]}"; do
        if [ -d "$job_dir" ]; then
            print_color $YELLOW "Cleaning job directory: $job_dir"
            
            # Remove old job output directories
            find "$job_dir" -name "*job*" -type d -mtime +$retention_days -exec rm -rf {} \; 2>/dev/null || true
            
            # Remove old log files
            find "$job_dir" -name "*.log" -type f -mtime +$retention_days -delete 2>/dev/null || true
            find "$job_dir" -name "*.out" -type f -mtime +$retention_days -delete 2>/dev/null || true
            find "$job_dir" -name "*.err" -type f -mtime +$retention_days -delete 2>/dev/null || true
            
            # Remove old checkpoint directories
            find "$job_dir" -name "*checkpoint*" -type d -mtime +$retention_days -exec rm -rf {} \; 2>/dev/null || true
        fi
    done
    
    print_color $GREEN "Job data cleanup completed"
}

# Function to cleanup conda/pip caches
cleanup_package_cache() {
    print_color $BLUE "Cleaning up package caches..."
    
    # Conda cache
    if command -v conda &> /dev/null; then
        print_color $YELLOW "Cleaning conda cache..."
        conda clean -y --all 2>/dev/null || true
    fi
    
    # Pip cache
    if command -v pip &> /dev/null; then
        print_color $YELLOW "Cleaning pip cache..."
        pip cache purge 2>/dev/null || true
    fi
    
    # APT cache (if available)
    if command -v apt-get &> /dev/null && [ "$(id -u)" -eq 0 ]; then
        print_color $YELLOW "Cleaning APT cache..."
        apt-get clean 2>/dev/null || true
        apt-get autoremove -y 2>/dev/null || true
    fi
    
    print_color $GREEN "Package cache cleanup completed"
}

# Function to cleanup old Python environments
cleanup_python_envs() {
    print_color $BLUE "Cleaning up old Python environments..."
    
    # Virtual environments
    local venv_dirs=(
        "$HOME/.virtualenvs"
        "$HOME/venvs"
        "$HOME/envs"
    )
    
    for venv_dir in "${venv_dirs[@]}"; do
        if [ -d "$venv_dir" ]; then
            print_color $YELLOW "Checking virtual environments in: $venv_dir"
            
            # Remove old unused virtual environments (older than 90 days)
            find "$venv_dir" -type d -mtime +90 -exec rm -rf {} \; 2>/dev/null || true
        fi
    done
    
    print_color $GREEN "Python environments cleanup completed"
}

# Function to find large files
find_large_files() {
    local size_threshold=${1:-100M}
    local search_path=${2:-$HOME}
    
    print_color $BLUE "Finding large files (>$size_threshold) in: $search_path"
    
    # Find large files
    find "$search_path" -type f -size +$size_threshold -exec ls -lh {} \; 2>/dev/null | \
        sort -k5 -hr | \
        head -20 | \
        awk '{printf "%-10s %s\n", $5, $9}'
    
    print_color $GREEN "Large files search completed"
}

# Function to cleanup old logs
cleanup_logs() {
    local retention_days=${1:-30}
    
    print_color $BLUE "Cleaning up old logs (older than $retention_days days)..."
    
    # Common log locations
    local log_dirs=(
        "/var/log"
        "$HOME/.cache"
        "$HOME/logs"
    )
    
    # Common log file patterns
    local log_patterns=(
        "*.log"
        "*.log.*"
        "*.out"
        "*.err"
        "syslog*"
        "messages*"
    )
    
    for log_dir in "${log_dirs[@]}"; do
        if [ -d "$log_dir" ]; then
            print_color $YELLOW "Cleaning log directory: $log_dir"
            
            for pattern in "${log_patterns[@]}"; do
                find "$log_dir" -name "$pattern" -type f -mtime +$retention_days -delete 2>/dev/null || true
            done
        fi
    done
    
    print_color $GREEN "Log cleanup completed"
}

# Function to show disk usage
show_disk_usage() {
    print_color $BLUE "Current disk usage:"
    
    # Overall disk usage
    df -h / /scratch /home 2>/dev/null || df -h
    
    # Directory sizes
    echo ""
    print_color $BLUE "Top 10 largest directories in $HOME:"
    du -sh "$HOME"/* 2>/dev/null | sort -hr | head -10
    
    # Scratch usage if available
    if [ -d "/scratch/$USER" ]; then
        echo ""
        print_color $BLUE "Scratch space usage:"
        du -sh "/scratch/$USER"/* 2>/dev/null | sort -hr | head -10
    fi
}

# Function to perform full cleanup
full_cleanup() {
    local retention_days=${1:-30}
    
    print_color $BLUE "Performing full cleanup (retention: $retention_days days)..."
    
    if confirm_action "This will remove temporary files, old job data, caches, and logs."; then
        cleanup_temp_files
        cleanup_job_data $retention_days
        cleanup_package_cache
        cleanup_python_envs
        cleanup_logs $retention_days
        
        print_color $GREEN "Full cleanup completed"
        show_disk_usage
    fi
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  temp              - Clean up temporary files"
        echo "  jobs [days]       - Clean up old job data (default: 30 days)"
        echo "  cache             - Clean up package caches"
        echo "  envs              - Clean up old Python environments"
        echo "  logs [days]       - Clean up old logs (default: 30 days)"
        echo "  large [size] [path] - Find large files (default: 100M, $HOME)"
        echo "  usage             - Show disk usage"
        echo "  full [days]       - Perform full cleanup (default: 30 days)"
        echo ""
        echo "Examples:"
        echo "  $0 temp                    # Clean temporary files"
        echo "  $0 jobs 7                  # Clean job data older than 7 days"
        echo "  $0 large 500M /scratch     # Find files >500M in /scratch"
        echo "  $0 full 14                 # Full cleanup with 14-day retention"
        exit 1
    fi
    
    command=$1
    option1=$2
    option2=$3
    
    case $command in
        temp)
            if confirm_action "This will remove temporary files and old logs."; then
                cleanup_temp_files
            fi
            ;;
        jobs)
            retention_days=${option1:-30}
            if confirm_action "This will remove job data older than $retention_days days."; then
                cleanup_job_data $retention_days
            fi
            ;;
        cache)
            if confirm_action "This will clean package caches."; then
                cleanup_package_cache
            fi
            ;;
        envs)
            if confirm_action "This will remove old Python environments."; then
                cleanup_python_envs
            fi
            ;;
        logs)
            retention_days=${option1:-30}
            if confirm_action "This will remove logs older than $retention_days days."; then
                cleanup_logs $retention_days
            fi
            ;;
        large)
            size_threshold=${option1:-100M}
            search_path=${option2:-$HOME}
            find_large_files $size_threshold $search_path
            ;;
        usage)
            show_disk_usage
            ;;
        full)
            retention_days=${option1:-30}
            full_cleanup $retention_days
            ;;
        *)
            print_color $RED "Unknown command: $command"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"