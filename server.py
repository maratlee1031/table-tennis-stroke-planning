"""HTTPS static server on :8443, so the phone can load controller.html.

    python server.py

Then open https://<PC-LAN-IP>:8443/controller.html on the phone and tap
"Advanced" -> "Proceed" past the self-signed certificate warning.

HTTPS is not optional here: browsers only expose the motion and orientation
sensors on a secure origin. The certificate is generated on first run
(OpenSSL if present, otherwise the Python cryptography package) and shared
with the game's WSS listener on :8766.

Note that certificate exceptions are per host **and port**, so accepting the
warning on :8443 does not cover :8766. If the WebSocket never connects, open
https://<PC-LAN-IP>:8766 once and accept it there too.
"""

import os, sys, ssl, http.server, socket, subprocess, shutil
from pathlib import Path

PORT = 8443
PEM_PATH = Path("selfsigned.pem")  # combined key+cert in one file

def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def gen_with_openssl(pem: Path) -> bool:
    exe = shutil.which("openssl")
    if not exe:
        # Try Git for Windows bundled OpenSSL
        candidate = r"C:\Program Files\Git\usr\bin\openssl.exe"
        if os.path.exists(candidate):
            exe = candidate
    if not exe:
        return False
    print(f"[i] Using OpenSSL at: {exe}")
    # One-file PEM (key + cert)
    cmd = [
        exe, "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(pem), "-out", str(pem),
        "-days", "365", "-nodes", "-subj", "/CN=localhost"
    ]
    subprocess.check_call(cmd)
    return True

def gen_with_python(pem: Path):
    """Generate a key + self-signed cert with the cryptography package."""
    try:
        import cryptography  # noqa: F401
    except ImportError:
        print("[i] installing 'cryptography'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])

    from datetime import datetime, timedelta, timezone
    from ipaddress import ip_address

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])

    # Subject Alternative Names: localhost plus the LAN IP, so the phone gets
    # a name match rather than an extra warning on top of the self-signed one
    alt_names = [x509.DNSName("localhost")]
    try:
        alt_names.append(x509.IPAddress(ip_address(local_ip())))
    except Exception:
        pass

    now = datetime.now(timezone.utc)     # utcnow() is deprecated in 3.12+
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )

    # Write combined PEM: private key + certificate
    pem.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        )
        + cert.public_bytes(serialization.Encoding.PEM)
    )
    print(f"[i] Generated self-signed certificate at: {pem}")

def ensure_cert(pem: Path):
    if pem.exists():
        return
    print("[i] No certificate found, generating self-signed certificate…")
    try:
        if gen_with_openssl(pem):
            return
        print("[i] OpenSSL not found; falling back to Python generation.")
        gen_with_python(pem)
    except subprocess.CalledProcessError as e:
        print("[!] Failed to run OpenSSL:", e)
        print("[i] Falling back to Python generation.")
        gen_with_python(pem)

def serve_https(port: int, pem: Path):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # keyfile and certfile are both the same combined PEM
    ctx.load_cert_chain(certfile=str(pem), keyfile=str(pem))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    ip = local_ip()
    print(f"\n[✓] HTTPS server running")
    print(f"    Local  : https://127.0.0.1:{port}/")
    print(f"    LAN    : https://{ip}:{port}/")
    print(f"    Folder : {Path.cwd()}\n")
    print("Note: On your phone, open the LAN URL above, tap 'Advanced' → 'Proceed' for the certificate warning.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    ensure_cert(PEM_PATH)
    serve_https(PORT, PEM_PATH)
