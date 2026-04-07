/**
 * API utility for making authenticated requests to StockSquad backend
 */

const API_URL = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';

/**
 * Get the current auth token from localStorage
 */
function getAuthToken() {
  return localStorage.getItem('stocksquad-auth-token');
}

/**
 * Create headers with authentication
 */
function createHeaders(additionalHeaders = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...additionalHeaders,
  };

  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
}

/**
 * Fetch with authentication
 */
export async function fetchWithAuth(endpoint, options = {}) {
  const url = `${API_URL}${endpoint}`;
  const headers = createHeaders(options.headers);

  const response = await fetch(url, {
    ...options,
    headers,
  });

  // Handle 401/403 errors
  if (response.status === 401 || response.status === 403) {
    const error = new Error('Authentication required or access denied');
    error.status = response.status;
    error.response = response;
    throw error;
  }

  return response;
}

/**
 * GET request with authentication
 */
export async function get(endpoint) {
  const response = await fetchWithAuth(endpoint, {
    method: 'GET',
  });

  if (!response.ok) {
    throw new Error(`GET ${endpoint} failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * POST request with authentication
 */
export async function post(endpoint, data) {
  const response = await fetchWithAuth(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`POST ${endpoint} failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * DELETE request with authentication
 */
export async function del(endpoint) {
  const response = await fetchWithAuth(endpoint, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`DELETE ${endpoint} failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Server-Sent Events with authentication
 * Returns EventSource-like object
 * Note: EventSource doesn't support custom headers, so we pass token as query param
 */
export function createAuthenticatedSSE(endpoint) {
  const token = getAuthToken();
  if (!token) {
    throw new Error('Authentication token required for SSE connection');
  }

  // Append token as query parameter since EventSource doesn't support custom headers
  const separator = endpoint.includes('?') ? '&' : '?';
  const url = `${API_URL}${endpoint}${separator}token=${encodeURIComponent(token)}`;

  return new EventSource(url);
}

export { API_URL };
