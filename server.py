import http.server
import os
import io
import hashlib
import pathlib
import mimetypes
import tempfile
import warnings
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode
from PIL import Image, features

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/avif", ".avif")

class ImagePlaceholderHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    PILLOW_FORMATS = {
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "png": "PNG",
        "webp": "WEBP",
        "gif": "GIF",
        "bmp": "BMP",
        "tif": "TIFF",
        "tiff": "TIFF",
        "avif": "AVIF",  # needs Pillow with AVIF
    }
    SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        *args,
        directory=None,
        cache_dir="cache",
        image_source_dir="high_quality_images",
        image_cache_limit=25,
        font_source_dir="fonts",
        **kwargs,
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        self.image_source_dir = os.path.abspath(image_source_dir)
        self.font_source_dir = pathlib.Path(os.path.abspath(font_source_dir))
        self.image_cache_limit = int(image_cache_limit)
        os.makedirs(self.cache_dir, exist_ok=True)

        base = pathlib.Path(self.image_source_dir)
        self.candidates = [
            str(p)
            for p in base.iterdir()
            if p.is_file() and p.suffix.lower() in self.SOURCE_EXTS
        ]

        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        super().end_headers()

    def log_message(self, format, *args):
        pass

    def send_head(self):
        parsed = urlparse(self.path)
        local_path = self.translate_path(parsed.path)

        # Existing files/directories: default behavior
        if os.path.exists(local_path):
            orig = self.path
            try:
                self.path = parsed.path  # strip query for parent
                return super().send_head()
            finally:
                self.path = orig

        # Font cache
        font_candidate = self.font_source_dir / parsed.path.lstrip('/')
        if os.path.exists(font_candidate):
            return self._serve_file(font_candidate)

        # Image placeholder request: ?width=...&height=...&type=...
        q = parse_qs(parsed.query)
        if {"width", "height", "type"}.issubset(q.keys()):
            return self._handle_special_image_request(parsed)

        self.send_error(404, "File not found")
        return None

    def do_GET(self):
        try:
            f = self.send_head()
            if f:
                try:
                    self.copyfile(f, self.wfile)
                finally:
                    f.close()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Client went away mid-response; nothing to do.
            pass

    def do_HEAD(self):
        try:
            f = self.send_head()
            if f:
                f.close()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Swallow disconnects during streaming
            pass

    def _handle_special_image_request(self, parsed):
        q = dict(parse_qs(parsed.query))
        try:
            width = int(q.get("width", [""])[0])
            height = int(q.get("height", [""])[0])
            img_ext = q.get("type", [""])[0].lower().lstrip(".")
        except Exception:
            self.send_error(400, "Invalid parameters")
            return None

        if width < 0 or height < 0 or not img_ext:
            self.send_error(400, "Invalid parameters")
            return None

        # 0x0 → empty body
        if width == 0 and height == 0:
            ctype = mimetypes.guess_type(f"dummy.{img_ext}")[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        pillow_fmt = self.PILLOW_FORMATS.get(img_ext)
        if not pillow_fmt:
            self.send_error(400, f"Unsupported type '{img_ext}'")
            self.close_connection = True
            return None

        if pillow_fmt == "AVIF" and not features.check("avif"):
            # Fallback to WEBP
            pillow_fmt = "WEBP"

        canonical_url = self._canonicalize_url(parsed)

        h = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:24]
        cached_name = f"{h}.{img_ext}"
        cached_path = os.path.join(self.cache_dir, cached_name)

        if os.path.exists(cached_path):
            return self._serve_file(cached_path)

        src = self._pick_source_image_for_url(canonical_url)
        if not src:
            self.send_error(500, f"No source images found in {self.image_source_dir}")
            self.close_connection = True
            return None

        tmp = tempfile.NamedTemporaryFile(
            dir=self.cache_dir,
            prefix=os.path.basename(cached_path) + ".",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()

        try:
            with Image.open(src) as im:
                if pillow_fmt in ("JPEG", "AVIF") and im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")

                # Only allow >0 x >0 (we handled 0x0 already)
                if width <= 0 or height <= 0:
                    self.send_error(400, "width and height must be positive (or both zero)")
                    return None

                im = im.resize((width, height), Image.Resampling.LANCZOS)

                save_kwargs = {}
                if pillow_fmt == "JPEG":
                    save_kwargs.update(quality=90, optimize=True, progressive=True)
                elif pillow_fmt in ("WEBP", "AVIF"):
                    save_kwargs.update(quality=90, method=6)

                im.save(tmp_path, format=pillow_fmt, **save_kwargs)

            try:
                os.replace(tmp_path, cached_path)
            except FileNotFoundError:
                # Our tmp was moved (or never created) — check if the final exists now.
                if not os.path.exists(cached_path):
                    raise
            # Touch mtime to "now" to keep LRU-ish behavior clear
            try:
                os.utime(cached_path, None)
            except Exception:
                pass

            # Purge if over the limit
            self._purge_cache_if_needed()

        except Exception as e:
            warnings.warn(f"Can't serve image request {type(e)}: {e}. Request was: {self.path}")
            self.send_error(500, f"Image processing error: {e}")
            self.close_connection = True
            return None

        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        return self._serve_file(cached_path)

    def _serve_file(self, path):
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(fs.st_size))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise

    def _canonicalize_url(self, parsed):
        """
        Make a deterministic, normalized representation of the URL for hashing:
        path + '?' + sorted query pairs (stable with duplicates).
        """
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        pairs.sort(key=lambda kv: (kv[0], kv[1]))
        canon_q = urlencode(pairs, doseq=True)
        return f"{parsed.path}?{canon_q}" if canon_q else parsed.path

    def _pick_source_image_for_url(self, canonical_url):
        """
        Deterministically choose a source image using a hash of the canonical URL.
        """
        non_query_url = canonical_url.split("?", 1)[0]

        digest = hashlib.sha256(non_query_url.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(self.candidates)
        return self.candidates[idx]

    def _purge_cache_if_needed(self):
        """
        If cache contains more than self.cache_limit files, delete the oldest
        until we are back at or below the limit. Ignores temp files.
        """
        try:
            entries = []
            for name in os.listdir(self.cache_dir):
                if name.endswith(".tmp") or name.endswith(".tmp-" + str(os.getpid())):
                    continue
                path = os.path.join(self.cache_dir, name)
                if os.path.isfile(path):
                    try:
                        st = os.stat(path)
                        entries.append((st.st_mtime, path))
                    except FileNotFoundError:
                        pass  # race with other threads/processes

            if len(entries) <= self.image_cache_limit:
                return

            # Oldest first
            entries.sort(key=lambda t: t[0])
            to_delete = len(entries) - self.image_cache_limit
            for _, path in entries[:to_delete]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except Exception:
                    continue
        except Exception:
            return


# Example server bootstrap
if __name__ == "__main__":
    from functools import partial
    VALIDATION_DATA_DIR = "static"
    CACHE_DIR = "cache"
    HQ_IMAGES_DIR = "high_quality_images"
    PORT = 3000

    Handler = partial(
        ImagePlaceholderHTTPRequestHandler,
        directory=VALIDATION_DATA_DIR,
        cache_dir=CACHE_DIR,
        source_dir=HQ_IMAGES_DIR,
        cache_limit=25,
    )

    with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Serving on http://localhost:{PORT}")
        httpd.serve_forever()