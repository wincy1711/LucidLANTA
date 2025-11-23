#!/bin/bash
# Data transfer utilities for LANTA cluster
# Provides efficient data transfer between storage locations

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

# Function to check if path exists
check_path() {
    local path=$1
    if [ ! -e "$path" ]; then
        print_color $RED "Path does not exist: $path"
        return 1
    fi
    return 0
}

# Function to calculate directory size
calculate_size() {
    local path=$1
    if [ -d "$path" ]; then
        du -sh "$path" | cut -f1
    elif [ -f "$path" ]; then
        ls -lh "$path" | awk '{print $5}'
    else
        echo "Unknown"
    fi
}

# Function to transfer data with progress
transfer_with_progress() {
    local source=$1
    local dest=$2
    local method=$3
    
    print_color $BLUE "Transferring: $source -> $dest"
    print_color $YELLOW "Method: $method"
    
    # Calculate source size
    local size=$(calculate_size "$source")
    print_color $GREEN "Source size: $size"
    
    # Start timer
    local start_time=$(date +%s)
    
    case $method in
        rsync)
            rsync -avh --info=progress2 "$source" "$dest"
            ;;
        cp)
            cp -r "$source" "$dest"
            ;;
        scp)
            scp -r "$source" "$dest"
            ;;
        *)
            print_color $RED "Unknown transfer method: $method"
            return 1
            ;;
    esac
    
    # Calculate transfer time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_color $GREEN "Transfer completed in ${duration}s"
}

# Function to compress before transfer
compress_and_transfer() {
    local source=$1
    local dest=$2
    local compression=${3:-gzip}
    
    print_color $BLUE "Compressing and transferring: $source"
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    local basename=$(basename "$source")
    local archive_name="${basename}_$(date +%Y%m%d_%H%M%S)"
    
    case $compression in
        gzip)
            archive_path="${temp_dir}/${archive_name}.tar.gz"
            print_color $YELLOW "Creating gzip archive..."
            tar -czf "$archive_path" -C "$(dirname "$source")" "$basename"
            ;;
        bzip2)
            archive_path="${temp_dir}/${archive_name}.tar.bz2"
            print_color $YELLOW "Creating bzip2 archive..."
            tar -cjf "$archive_path" -C "$(dirname "$source")" "$basename"
            ;;
        xz)
            archive_path="${temp_dir}/${archive_name}.tar.xz"
            print_color $YELLOW "Creating xz archive..."
            tar -cJf "$archive_path" -C "$(dirname "$source")" "$basename"
            ;;
        zip)
            archive_path="${temp_dir}/${archive_name}.zip"
            print_color $YELLOW "Creating zip archive..."
            cd "$(dirname "$source")" && zip -r "$archive_path" "$basename"
            ;;
        *)
            print_color $RED "Unknown compression method: $compression"
            rm -rf "$temp_dir"
            return 1
            ;;
    esac
    
    # Transfer archive
    print_color $YELLOW "Transferring archive..."
    transfer_with_progress "$archive_path" "$dest" "rsync"
    
    # Cleanup
    rm -rf "$temp_dir"
    print_color $GREEN "Compression and transfer completed"
}

# Function to sync directories
sync_directories() {
    local source=$1
    local dest=$2
    local exclude_pattern=${3:-""}
    
    print_color $BLUE "Syncing directories: $source -> $dest"
    
    # Build rsync command
    local rsync_cmd="rsync -avh --info=progress2 --delete"
    
    # Add exclude pattern if provided
    if [ -n "$exclude_pattern" ]; then
        rsync_cmd="$rsync_cmd --exclude='$exclude_pattern'"
    fi
    
    # Add source and destination
    rsync_cmd="$rsync_cmd '$source/' '$dest/'"
    
    print_color $YELLOW "Running: $rsync_cmd"
    eval "$rsync_cmd"
    
    print_color $GREEN "Directory sync completed"
}

# Function to backup important data
backup_data() {
    local source=$1
    local backup_dir=$2
    local retention_days=${3:-30}
    
    print_color $BLUE "Creating backup: $source -> $backup_dir"
    
    # Create backup directory with timestamp
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="${backup_dir}/backup_${timestamp}"
    
    mkdir -p "$backup_path"
    
    # Copy data
    print_color $YELLOW "Copying data to backup location..."
    cp -r "$source" "$backup_path/"
    
    # Create metadata file
    cat > "${backup_path}/backup_metadata.txt" << EOF
Backup created: $(date)
Source: $source
Backup location: $backup_path
Size: $(du -sh "$backup_path" | cut -f1)
EOF
    
    # Clean up old backups
    if [ "$retention_days" -gt 0 ]; then
        print_color $YELLOW "Cleaning up backups older than $retention_days days..."
        find "$backup_dir" -name "backup_*" -type d -mtime +$retention_days -exec rm -rf {} \; 2>/dev/null || true
    fi
    
    print_color $GREEN "Backup completed: $backup_path"
}

# Function to verify transfer integrity
verify_transfer() {
    local source=$1
    local dest=$2
    
    print_color $BLUE "Verifying transfer integrity..."
    
    # Calculate checksums
    print_color $YELLOW "Calculating source checksum..."
    local source_hash=$(find "$source" -type f -exec md5sum {} \; | sort | md5sum | cut -d' ' -f1)
    
    print_color $YELLOW "Calculating destination checksum..."
    local dest_hash=$(find "$dest" -type f -exec md5sum {} \; | sort | md5sum | cut -d' ' -f1)
    
    # Compare
    if [ "$source_hash" = "$dest_hash" ]; then
        print_color $GREEN "Transfer verification PASSED"
        return 0
    else
        print_color $RED "Transfer verification FAILED"
        print_color $RED "Source hash: $source_hash"
        print_color $RED "Destination hash: $dest_hash"
        return 1
    fi
}

# Function to show transfer statistics
show_transfer_stats() {
    local path=$1
    
    print_color $BLUE "Transfer statistics for: $path"
    
    if [ -d "$path" ]; then
        # Directory statistics
        local file_count=$(find "$path" -type f | wc -l)
        local dir_count=$(find "$path" -type d | wc -l)
        local total_size=$(du -sh "$path" | cut -f1)
        
        print_color $GREEN "Files: $file_count"
        print_color $GREEN "Directories: $dir_count"
        print_color $GREEN "Total size: $total_size"
        
        # Largest files
        print_color $YELLOW "Largest files:"
        find "$path" -type f -exec ls -lh {} \; | sort -k5 -hr | head -10 | awk '{print $9 " " $5}'
        
    elif [ -f "$path" ]; then
        # File statistics
        local size=$(ls -lh "$path" | awk '{print $5}')
        local modified=$(stat -c %y "$path")
        
        print_color $GREEN "Size: $size"
        print_color $GREEN "Last modified: $modified"
    fi
}

# Main function
main() {
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <command> <source> [destination] [options]"
        echo ""
        echo "Commands:"
        echo "  transfer <source> <dest> [method]     - Transfer data (rsync, cp, scp)"
        echo "  compress <source> <dest> [compression] - Compress and transfer"
        echo "  sync <source> <dest> [exclude]        - Sync directories"
        echo "  backup <source> <backup_dir> [days]   - Create backup with retention"
        echo "  verify <source> <dest>                - Verify transfer integrity"
        echo "  stats <path>                          - Show transfer statistics"
        echo ""
        echo "Examples:"
        echo "  $0 transfer /data /scratch rsync"
        echo "  $0 compress /data /scratch gzip"
        echo "  $0 sync /data /backup '*.tmp'"
        echo "  $0 backup /project /backups 30"
        echo "  $0 verify /source /destination"
        echo "  $0 stats /data"
        exit 1
    fi
    
    command=$1
    source=$2
    dest=$3
    option=$4
    
    # Check source path
    if ! check_path "$source"; then
        exit 1
    fi
    
    case $command in
        transfer)
            method=${option:-rsync}
            transfer_with_progress "$source" "$dest" "$method"
            ;;
        compress)
            compression=${option:-gzip}
            compress_and_transfer "$source" "$dest" "$compression"
            ;;
        sync)
            exclude_pattern=$option
            sync_directories "$source" "$dest" "$exclude_pattern"
            ;;
        backup)
            retention_days=${option:-30}
            backup_data "$source" "$dest" "$retention_days"
            ;;
        verify)
            verify_transfer "$source" "$dest"
            ;;
        stats)
            show_transfer_stats "$source"
            ;;
        *)
            print_color $RED "Unknown command: $command"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"