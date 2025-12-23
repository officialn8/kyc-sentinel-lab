#!/usr/bin/env python3
"""
Set up Modal secrets for R2 access.

This script creates a Modal secret named 'r2-credentials' containing
the R2/MinIO credentials needed for the video extraction worker.

Usage:
    # Set environment variables first:
    export R2_ENDPOINT="https://your-account.r2.cloudflarestorage.com"
    export R2_ACCESS_KEY="your-access-key"
    export R2_SECRET_KEY="your-secret-key"
    export R2_BUCKET="kyc-sentinel-media"

    # Then run:
    python scripts/setup_modal_secrets.py

For local development with MinIO:
    export R2_ENDPOINT="http://localhost:9000"
    export R2_ACCESS_KEY="minioadmin"
    export R2_SECRET_KEY="minioadmin"
    export R2_BUCKET="kyc-sentinel-media"
"""

import os
import sys


def main():
    """Create or update the Modal secret for R2 credentials."""
    try:
        import modal
    except ImportError:
        print("Error: Modal is not installed. Run: pip install modal")
        sys.exit(1)

    # Get credentials from environment
    r2_endpoint = os.environ.get("R2_ENDPOINT")
    r2_access_key = os.environ.get("R2_ACCESS_KEY")
    r2_secret_key = os.environ.get("R2_SECRET_KEY")
    r2_bucket = os.environ.get("R2_BUCKET", "kyc-sentinel-media")

    # Validate
    missing = []
    if not r2_endpoint:
        missing.append("R2_ENDPOINT")
    if not r2_access_key:
        missing.append("R2_ACCESS_KEY")
    if not r2_secret_key:
        missing.append("R2_SECRET_KEY")

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set these environment variables and try again.")
        print("\nExample for local MinIO:")
        print('  export R2_ENDPOINT="http://localhost:9000"')
        print('  export R2_ACCESS_KEY="minioadmin"')
        print('  export R2_SECRET_KEY="minioadmin"')
        print('  export R2_BUCKET="kyc-sentinel-media"')
        sys.exit(1)

    # Create the secret
    print("Creating Modal secret 'r2-credentials'...")
    print(f"  R2_ENDPOINT: {r2_endpoint}")
    print(f"  R2_ACCESS_KEY: {r2_access_key[:4]}...{r2_access_key[-4:]}")
    print(f"  R2_SECRET_KEY: {'*' * 20}")
    print(f"  R2_BUCKET: {r2_bucket}")

    try:
        # Use modal CLI to create the secret
        secret = modal.Secret.from_dict(
            {
                "R2_ENDPOINT": r2_endpoint,
                "R2_ACCESS_KEY": r2_access_key,
                "R2_SECRET_KEY": r2_secret_key,
                "R2_BUCKET": r2_bucket,
            }
        )
        
        # The secret is created when we deploy the app
        # For now, we'll use the modal CLI to create it
        print("\nNote: To create the secret in Modal, run:")
        print("  modal secret create r2-credentials \\")
        print(f'    R2_ENDPOINT="{r2_endpoint}" \\')
        print(f'    R2_ACCESS_KEY="{r2_access_key}" \\')
        print(f'    R2_SECRET_KEY="<your-secret>" \\')
        print(f'    R2_BUCKET="{r2_bucket}"')
        print("\nOr use the Modal dashboard: https://modal.com/secrets")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\nDone!")


def create_secret_interactive():
    """Interactive mode to create the secret."""
    try:
        import modal
    except ImportError:
        print("Error: Modal is not installed. Run: pip install modal")
        sys.exit(1)

    print("Modal R2 Credentials Setup")
    print("=" * 40)
    print()

    # Get user input
    default_endpoint = "http://localhost:9000"
    r2_endpoint = input(f"R2 Endpoint [{default_endpoint}]: ").strip() or default_endpoint

    default_access = "minioadmin"
    r2_access_key = input(f"R2 Access Key [{default_access}]: ").strip() or default_access

    default_secret = "minioadmin"
    r2_secret_key = input(f"R2 Secret Key [{default_secret}]: ").strip() or default_secret

    default_bucket = "kyc-sentinel-media"
    r2_bucket = input(f"R2 Bucket [{default_bucket}]: ").strip() or default_bucket

    print()
    print("Creating secret with:")
    print(f"  Endpoint: {r2_endpoint}")
    print(f"  Access Key: {r2_access_key}")
    print(f"  Bucket: {r2_bucket}")
    print()

    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Aborted.")
        sys.exit(0)

    print("\nTo create the secret, run:")
    print("  modal secret create r2-credentials \\")
    print(f'    R2_ENDPOINT="{r2_endpoint}" \\')
    print(f'    R2_ACCESS_KEY="{r2_access_key}" \\')
    print(f'    R2_SECRET_KEY="{r2_secret_key}" \\')
    print(f'    R2_BUCKET="{r2_bucket}"')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        create_secret_interactive()
    else:
        main()


