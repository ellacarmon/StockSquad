"""
JWT token service for secure session management.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt


class JWTService:
    """Service for creating and validating JWT tokens."""

    def __init__(self):
        """Initialize JWT service."""
        self.secret_key = os.getenv("JWT_SECRET_KEY", "")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days default

        if not self.secret_key:
            # Generate a random secret for development
            import secrets
            self.secret_key = secrets.token_urlsafe(32)
            print(f"⚠️  WARNING: JWT_SECRET_KEY not set. Using temporary key (NOT FOR PRODUCTION!)")
            print(f"    Set JWT_SECRET_KEY in your environment for production use.")
        else:
            print(f"✅ JWT service initialized (token expiry: {self.access_token_expire_minutes} minutes)")

    def create_access_token(self, email: str) -> str:
        """
        Create a JWT access token for the given email.

        Args:
            email: User's email address (verified)

        Returns:
            JWT token string
        """
        expires_at = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": email,  # Subject (user identifier)
            "email": email,
            "iat": datetime.utcnow(),  # Issued at
            "exp": expires_at,  # Expiration
            "type": "access"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        print(f"🔑 Created JWT token for {email} (expires: {expires_at.isoformat()})")
        return token

    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify a JWT token and extract the email.

        Args:
            token: JWT token string

        Returns:
            Email if valid, None if invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            email = payload.get("email")
            if not email:
                print("❌ JWT token missing email claim")
                return None

            print(f"✅ JWT token verified for {email}")
            return email

        except JWTError as e:
            print(f"❌ JWT verification failed: {e}")
            return None

    def decode_token_unsafe(self, token: str) -> Optional[dict]:
        """
        Decode token without verification (for debugging only).

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except:
            return None


# Global JWT service instance
jwt_service = JWTService()
