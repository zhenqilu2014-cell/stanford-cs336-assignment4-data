import subprocess
import timeit
from pathlib import Path

gstart = timeit.default_timer()

prefix = "https://data.commoncrawl.org/"
fin = open("wet_paths.txt", "r")
MAX_ITER = 2000

for count in range(MAX_ITER):
    url_suffix = fin.readline().rstrip('\n')
    output_name = f"cs336_data/data/common_crawl_raw/{url_suffix.split('/')[-1]}"
    if Path(output_name).exists():
        continue
    url = f"{prefix}{url_suffix}"
    command = ["wget", "-O", output_name, url]
    subprocess.run(command)
    if (count + 1) % 30 == 0:
        print(f"Number of files processed: {count + 1}")

fin.close()

gstop = timeit.default_timer()
print(f"Total Execution Time: {(gstop - gstart)/60:2f} minutes")