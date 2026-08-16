"""Range 分段并行下载（绕过单连接限速）。用法：python parallel_download.py <url> <out> [N=8]"""
import os
import sys
import threading

import requests

URL = sys.argv[1]
OUT = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def get_size(url):
    r = requests.head(url, timeout=30, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def dl(i, size, chunk, errors):
    start = i * chunk
    end = size - 1 if i == N - 1 else (i + 1) * chunk - 1
    tmp = f"{OUT}.part{i}"
    try:
        r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"},
                         stream=True, timeout=120, allow_redirects=True)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
        print(f"段{i} 完成 {os.path.getsize(tmp)} 字节", flush=True)
    except Exception as e:
        errors.append((i, str(e)))
        print(f"段{i} 失败: {e}", flush=True)


def main():
    size = get_size(URL)
    print(f"总大小 {size} 字节，{N} 段并行", flush=True)
    chunk = size // N
    errors = []
    threads = [threading.Thread(target=dl, args=(i, size, chunk, errors)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print("有失败段，需重试:", errors)
        sys.exit(1)
    with open(OUT, "wb") as out:
        for i in range(N):
            with open(f"{OUT}.part{i}", "rb") as p:
                out.write(p.read())
    for i in range(N):
        os.remove(f"{OUT}.part{i}")
    print(f"合并完成 {os.path.getsize(OUT)} 字节")


if __name__ == "__main__":
    main()
