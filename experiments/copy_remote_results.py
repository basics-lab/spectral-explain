import os
import paramiko
from scp import SCPClient

def create_ssh_client(hostname, username, password=None, key_file=None, port=22):
    """Create an SSH client."""
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if key_file:
        ssh.connect(hostname, port=port, username=username, key_filename=key_file)
    else:
        ssh.connect(hostname, port=port, username=username, password=password)
    return ssh


def list_remote_subfolders(ssh_client, remote_folder):
    """List sub-folders in the remote folder."""
    stdin, stdout, stderr = ssh_client.exec_command(f"ls -d {remote_folder}/*/")
    subfolders = [line.strip() for line in stdout.readlines()]
    print(subfolders)
    return subfolders


def copy_remote_subfolder_to_local(ssh_client, remote_subfolder, local_base_folder):
    """Copy a single remote subfolder to the local folder."""
    folder_name = os.path.basename(remote_subfolder.rstrip("/"))
    print(f'folder_name: {folder_name}')
    print(f'remote_subfolder: {remote_subfolder}')
    print(f'local_base_folder: {local_base_folder}')
    #local_subfolder = local_base_folder#os.path.join(local_base_folder, folder_name)
    local_subfolder = os.path.join(local_base_folder, folder_name)
    print(f'local_subfolder: {local_subfolder}')

    if os.path.exists(local_subfolder):
        print(f"Folder '{local_subfolder}' already exists locally. Skipping copy.")
        return

    print(f"Copying '{remote_subfolder}' from remote server to '{local_subfolder}'...")
    os.makedirs(local_base_folder, exist_ok=True)
    #temp_folder = os.path.join(local_base_folder, f"temp_{folder_name}")
    #os.makedirs(temp_folder, exist_ok=True)
    with SCPClient(ssh_client.get_transport()) as scp:
        scp.get(f'{remote_subfolder}', local_subfolder, recursive=True)
        print(f"Copy of '{remote_subfolder}' completed.")
        #for item in os.listdir(remote_subfolder):
        #    item_path = os.path.join(remote_subfolder, item)
        #    target_path = os.path.join(temp_folder, item)
        #    os.rename(item_path, target_path)
            #print(f"Copy of '{remote_subfolder}' completed.")
    #os.rmdir(temp_folder)

def main():
    # Configuration
    hostname = "bridges2.psc.edu"  # Replace with your remote server's hostname or IP
    username = "aagarwa6"      # Replace with your username
    password = None                # Set this if you use a password
    key_file = None#"~/.ssh/id_rsa"      # Replace with your SSH key file path, or None if not used
    remote_folder = "/jet/home/aagarwa6/spectral-explain/experiments/results/drop"  # Replace with the remote folder path
    local_folder = "experiments/results/drop"    # Replace with the local folder path

    # Connect to the remote server
    try:
        ssh_client = create_ssh_client(hostname, username, password, key_file)

        # Get list of sub-folders from the remote folder
        remote_subfolders = list_remote_subfolders(ssh_client, remote_folder)
        print(f"Found {len(remote_subfolders)} sub-folders to process.")

        # Copy each sub-folder individually if not already present locally
        for remote_subfolder in remote_subfolders:
            copy_remote_subfolder_to_local(ssh_client, remote_subfolder, local_folder)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ssh_client.close()

if __name__ == "__main__":
    main()
