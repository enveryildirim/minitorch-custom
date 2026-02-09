import os
import urllib.request
import gzip
import shutil

DATA_DIR = "project/data"
# Use a reliable mirror
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    for filename in FILES:
        filepath = os.path.join(DATA_DIR, filename)
        url = BASE_URL + filename

        if not os.path.exists(filepath):
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue

        # Extract
        with gzip.open(filepath, "rb") as f_in:
            extracted_filename = filename.replace(".gz", "")
            extracted_filepath = os.path.join(DATA_DIR, extracted_filename)
            if not os.path.exists(extracted_filepath):
                print(f"Extracting {filename}...")
                with open(extracted_filepath, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    download_and_extract()
