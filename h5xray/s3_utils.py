import s3fs
import h5py

def parse_s3_url(s3_url):
    # Parse the S3 URL to extract bucket name and key
    parts = s3_url.replace('s3://', '').split('/')
    bucket_name = parts[0]
    key = '/'.join(parts[1:])
    return bucket_name, key

def open_hdf5_file_from_s3(s3_url):
    # Parse the S3 URL to extract bucket name and key
    bucket_name, key = parse_s3_url(s3_url)

    # Create an s3fs filesystem
    fs = s3fs.S3FileSystem(anon=False)  # Set anon=False if using AWS credentials

    # Open the HDF5 file directly from S3
    with fs.open(f"{bucket_name}/{key}", "rb") as s3_file:
        # Wrap the S3 file in an h5py File object
        hdf5_file = h5py.File(s3_file, "r")

    return hdf5_file
