"""Async client for a new-api instance's admin user-management API.

Authentication uses the admin's system access token (``Authorization:
Bearer``) together with the ``new-api-user`` header, which is how the web
console's own requests are signed; no session or Cloudflare cookies needed.
"""

from __future__ import annotations

import aiohttp

# new-api prices quota at 500000 units per US dollar.
QUOTA_PER_USD = 500000


class NewApiError(Exception):
    """Raised when a new-api call cannot be completed."""


class NewApiClient:
    """Minimal admin client: look users up and credit their quota."""

    def __init__(
        self,
        base_url: str,
        admin_user_id: int,
        access_token: str,
        request_timeout: float = 30.0,
    ) -> None:
        base = base_url.strip().rstrip("/")
        if not base:
            raise NewApiError("new-api base URL is not configured")
        if admin_user_id < 1:
            raise NewApiError("new-api admin user id is not configured")
        token = access_token.strip()
        if not token:
            raise NewApiError("new-api access token is not configured")
        self._base = base
        self._headers = {
            "Authorization": f"Bearer {token}",
            "new-api-user": str(admin_user_id),
        }
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Create the shared session on first use.

        Returns:
            The session bound to the running event loop.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        """Send one JSON request and unwrap new-api's success envelope.

        Args:
            method: HTTP method, e.g. ``GET`` or ``POST``.
            path: Path below the base URL, e.g. ``/api/user/147``.
            payload: Optional JSON body.

        Returns:
            The parsed response body.

        Raises:
            NewApiError: On network errors, non-JSON replies such as a
                Cloudflare challenge page, or a ``success: false`` envelope
                (its message is relayed).
        """
        session = await self._get_session()
        try:
            async with session.request(
                method,
                f"{self._base}{path}",
                headers=self._headers,
                json=payload,
            ) as resp:
                try:
                    data = await resp.json(content_type=None)
                except (ValueError, aiohttp.ContentTypeError):
                    raise NewApiError(
                        f"unexpected response (HTTP {resp.status})",
                    ) from None
        except NewApiError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise NewApiError(f"request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise NewApiError("unexpected response body")
        if not data.get("success"):
            message = str(data.get("message") or "").strip()
            raise NewApiError(message or "request rejected")
        return data

    async def get_user(self, user_id: int) -> dict:
        """Fetch a new-api user record.

        Args:
            user_id: The numeric new-api user id.

        Returns:
            The ``data`` object describing the user (username, quota, ...).

        Raises:
            NewApiError: If the user does not exist or the lookup failed.
        """
        data = await self._request("GET", f"/api/user/{user_id}")
        return data.get("data") or {}

    async def add_quota(self, user_id: int, quota: int) -> None:
        """Credit a user's balance with additive quota.

        Args:
            user_id: The numeric new-api user id.
            quota: Amount to add; ``QUOTA_PER_USD`` units equal one US dollar.

        Raises:
            NewApiError: If the credit was rejected or the call failed.
        """
        await self._request(
            "POST",
            "/api/user/manage",
            {"id": user_id, "action": "add_quota", "mode": "add", "value": quota},
        )

    async def close(self) -> None:
        """Release the HTTP session if one was opened."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
