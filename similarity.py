import argparse
import json
import subprocess
import sys
import warnings

from playwright.sync_api import sync_playwright, TimeoutError

import torch
from PIL import Image
import PIL
import torchvision.transforms as transforms
import numpy as np
import lpips
import os

from functools import partial
from pathlib import Path
import http.server

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
WEB_SERVER_PORT = 3000
VALIDATION_DATA_DIR = 'data-rb-validate'

server_thread = None

def remove_alpha(image):
    if image.shape[0] == 4:
        return image[:3, :, :]
    return image

def calculate_mse(image1, image2):
    return ((image1 - image2) ** 2).mean().item()

def resize_image(image):
    return torch.nn.functional.avg_pool2d(image, 2)

def calculate_similarity(image1, image2):
    errors = []
    while True:
        mse = calculate_mse(image1, image2)

        errors.append(mse)

        _, h, w = image1.size()
        if h == 1 or w == 1:
            break

        image1 = resize_image(image1)
        image2 = resize_image(image2)

    average_mse = np.mean(errors)
    sim = 1 - average_mse

    return float(sim)

def calculate_perceptual_loss(image1, image2):
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
    loss_fn_alex.to(image1.device)

    # Normalize to [-1, 1]
    image1 = (image1 - 0.5) * 2
    image2 = (image2 - 0.5) * 2

    loss = loss_fn_alex(image1, image2)

    return loss.squeeze().item()

def metrics(image1_path, image2_path):
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        # temporarily force CPU to save GPU memory
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.ToTensor()

    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if image1.size == image2.size:
        new_image2 = Image.new("RGB", image1.size, (255, 255, 255))
        new_image2.paste(image2, (0, 0))
        image2 = new_image2.crop((0, 0, image1.size[0], image1.size[1]))

    image1 = remove_alpha(transform(image1)).to(device)
    image2 = remove_alpha(transform(image2)).to(device)

    similarity = calculate_similarity(image1, image2)
    perceptual_loss = calculate_perceptual_loss(image1, image2)

    return {
        'similarity': similarity,
        'perceptual_loss': perceptual_loss,
    }


def take_screenshot(url, path, viewport_width, viewport_height, max_retries=3):
    os.environ["PW_TEST_SCREENSHOT_NO_FONTS_READY"] = "1"
    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--disable-gpu"])
                context = browser.new_context(viewport={'width': viewport_width, 'height': viewport_height},
                                              service_workers="block")
                page = context.new_page()
                try:
                    page.goto(url, wait_until="networkidle", timeout=10000)
                    page.wait_for_timeout(1000)
                except TimeoutError:
                    warnings.warn(f"Timeout loading page: {url}. Trying without waiting for networkidle.")
                    page.goto(url, wait_until="domcontentloaded", timeout=10000)
                    page.wait_for_timeout(5000)

                page.screenshot(path=path)
                context.close()
                browser.close()
                return  # success
        except Exception as e:
            if attempt < max_retries - 1:
                warnings.warn(f"Screenshot attempt {attempt + 1} failed: {e}. Retrying...")
                import time
                time.sleep(1)
            else:
                raise

import socketserver
import threading
import os

stop_server_flag = threading.Event()

def start_server():
    from .server import ImagePlaceholderHTTPRequestHandler
    Handler = partial(
        ImagePlaceholderHTTPRequestHandler,
        directory=VALIDATION_DATA_DIR,
        cache_dir='cache',
        image_source_dir=Path(__file__).resolve().parent / 'DIV2K_valid_HR',
        font_source_dir=Path(__file__).resolve().parent / 'fonts',
        image_cache_limit=500,
    )

    while not stop_server_flag.is_set():
        try:
            with http.server.ThreadingHTTPServer(("0.0.0.0", WEB_SERVER_PORT), Handler) as httpd:
                httpd.allow_reuse_address = True
                httpd.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 48:  # Address already in use
                print(f"Server already running. Waiting for stop signal.")
                # Wait until stop flag is set
                while not stop_server_flag.is_set():
                    import time
                    time.sleep(1)
                break  # Exit the outer loop
            else:
                print(f"Server error: {e}, retrying in 1 second")
                import time
                time.sleep(1)
        except Exception as e:
            print(f"Server error: {e}, retrying in 1 second")
            import time
            time.sleep(1)

def calculate_metrics(predicted_markup, expected_markup, viewport_width, viewport_height):
    global server_thread

    import uuid
    uuid = uuid.uuid4().hex

    predicted_html_path = os.path.join(VALIDATION_DATA_DIR, f'predicted{uuid}.html')
    expected_html_path = os.path.join(VALIDATION_DATA_DIR, f'expected{uuid}.html')

    with open(predicted_html_path, 'w') as file:
        file.write(predicted_markup)

    with open(expected_html_path, 'w') as file:
        file.write(expected_markup)

    predicted_screenshot_path = os.path.join(VALIDATION_DATA_DIR, f'predicted{uuid}.png')
    expected_screenshot_path = os.path.join(VALIDATION_DATA_DIR, f'expected{uuid}.png')

    try:
        if server_thread is None or not server_thread.is_alive():
            server_thread = threading.Thread(target=start_server,)
            server_thread.daemon = True
            server_thread.start()

        script_path = os.path.abspath(__file__)

        result = subprocess.run(
            [sys.executable, script_path,
                'http://localhost:' + str(WEB_SERVER_PORT) + '/' + os.path.basename(predicted_html_path),
                'http://localhost:' + str(WEB_SERVER_PORT) + '/' + os.path.basename(expected_html_path),
                predicted_screenshot_path,
                expected_screenshot_path,
                str(viewport_width),
                str(viewport_height),
             ],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output as a float
        output = result.stdout.strip()

        metrics = json.loads(output)
        metrics['predicted_screenshot_path'] = predicted_screenshot_path
        metrics['expected_screenshot_path'] = expected_screenshot_path
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e.stderr}")
        return None

    except json.decoder.JSONDecodeError:
        print(f"Output is not a json: {output}")
        return None

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('predicted_url', type=str,)
    parser.add_argument('expected_url', type=str,)
    parser.add_argument('predicted_screenshot_path', type=str,)
    parser.add_argument('expected_screenshot_path', type=str,)
    parser.add_argument('viewport_width', type=int,)
    parser.add_argument('viewport_height', type=int,)

    args = parser.parse_args()

    take_screenshot(args.predicted_url, args.predicted_screenshot_path, args.viewport_width, args.viewport_height)
    take_screenshot(args.expected_url, args.expected_screenshot_path, args.viewport_width, args.viewport_height)

    metrics = metrics(args.predicted_screenshot_path, args.expected_screenshot_path)

    print(json.dumps(metrics))

