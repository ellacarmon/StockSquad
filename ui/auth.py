"""
Authorization middleware using Permit.io
Provides user authentication and permission management for StockSquad API.
"""

import os
from typing import Optional
from fastapi import Header, HTTPException, status
from permit import Permit
from ui.jwt_service import jwt_service


class PermitAuth:
    """Permit.io authorization handler."""

    def __init__(self):
        """Initialize Permit.io client."""
        api_key = os.getenv("PERMIT_IO_API_KEY", "")
        pdp_url = os.getenv("PERMIT_IO_PDP_URL", "https://cloudpdp.api.permit.io")
        self.skip_auth = os.getenv("SKIP_AUTH", "false").lower() == "true"

        if self.skip_auth:
            print("🚀 Authorization skipped (SKIP_AUTH=true). Defaulting to full access.")
            self.permit = None
            self.enabled = False
            return

        if not api_key:
            print("WARNING: PERMIT_IO_API_KEY not set. Authorization will be disabled.")
            self.permit = None
            self.enabled = False
        else:
            self.permit = Permit(
                token=api_key,
                pdp=pdp_url
            )
            self.enabled = True
            print(f"Permit.io authorization enabled (PDP: {pdp_url})")

    async def check_permission(
        self,
        user_id: str,
        action: str,
        resource: str = "analysis"
    ) -> bool:
        """
        Check if user has permission to perform action.

        Args:
            user_id: Unique identifier for the user (email, ID, etc.)
            action: The action to check (e.g., "create", "read", "delete")
            resource: The resource type (default: "analysis")

        Returns:
            True if user is authorized, False otherwise
        """
        if not self.enabled:
            # If Permit.io is not configured, allow all requests (development mode)
            return True

        try:
            print(f"🔍 Checking Permit.io: user={user_id}, action={action}, resource={resource}")
            allowed = await self.permit.check(
                user=user_id,
                action=action,
                resource=resource
            )
            print(f"✅ Permit.io result: {'ALLOWED' if allowed else 'DENIED'}")
            return allowed
        except Exception as e:
            print(f"❌ Permit.io check failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            # Fail closed - deny access on errors
            return False

    async def require_permission(
        self,
        user_id: Optional[str],
        action: str,
        resource: str = "analysis"
    ):
        """
        Require user to have permission, raise HTTPException if not authorized.

        Args:
            user_id: User identifier from header
            action: The action to check
            resource: The resource type

        Raises:
            HTTPException: 401 if no user_id, 403 if not authorized
        """
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide X-User-Id header.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        allowed = await self.check_permission(user_id, action, resource)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{user_id}' is not authorized to {action} {resource}.",
            )


# Global instance
permit_auth = PermitAuth()


async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> Optional[str]:
    """
    FastAPI dependency to extract user from JWT token or X-User-Id header.

    Checks Authorization header first (Bearer token), then falls back to X-User-Id.

    Usage:
        @app.post("/api/something")
        async def my_endpoint(user_id: str = Depends(get_current_user)):
            # user_id will be None if neither auth method provided
            pass
    """
    # Try JWT token first
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        email = jwt_service.verify_token(token)
        if email:
            return email

    # Fall back to X-User-Id header (for backward compatibility)
    if x_user_id:
        return x_user_id

    # If SKIP_AUTH is enabled, use a default user
    if os.getenv("SKIP_AUTH", "false").lower() == "true":
        return "dev@stocksquad.app"

    return None


async def require_auth(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> str:
    """
    FastAPI dependency that requires authentication.

    Accepts either JWT token (preferred) or X-User-Id header.

    Usage:
        @app.post("/api/something")
        async def my_endpoint(user_id: str = Depends(require_auth)):
            # Will raise 401 if neither auth method provided
            pass
    """
    # Try JWT token first
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        email = jwt_service.verify_token(token)
        if email:
            return email
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Fall back to X-User-Id header
    if x_user_id:
        return x_user_id

    # If SKIP_AUTH is enabled, use a default user
    if os.getenv("SKIP_AUTH", "false").lower() == "true":
        return "dev@stocksquad.app"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide Authorization header with Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_create_permission(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> str:
    """
    FastAPI dependency that requires 'create' permission on 'analysis' resource.

    Usage:
        @app.post("/api/analyze")
        async def analyze(user_id: str = Depends(require_create_permission)):
            # Will raise 401/403 if not authorized
            pass
    """
    # Get user_id using the same logic as require_auth
    user_id = await require_auth(authorization, x_user_id)
    await permit_auth.require_permission(user_id, "create", "analysis")
    return user_id


async def require_delete_permission(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> str:
    """
    FastAPI dependency that requires 'delete' permission.

    Usage:
        @app.delete("/api/something/{id}")
        async def delete(id: str, user_id: str = Depends(require_delete_permission)):
            # Will raise 401/403 if not authorized
            pass
    """
    # Get user_id using the same logic as require_auth
    user_id = await require_auth(authorization, x_user_id)
    await permit_auth.require_permission(user_id, "delete", "analysis")
    return user_id
