import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [userId, setUserId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for stored token on mount
    const storedToken = localStorage.getItem('stocksquad-auth-token');
    const storedUserId = localStorage.getItem('stocksquad-user-id');

    if (storedToken && storedUserId) {
      setToken(storedToken);
      setUserId(storedUserId);
    }
    setIsLoading(false);
  }, []);

  const login = (authToken, email) => {
    setToken(authToken);
    setUserId(email);
    localStorage.setItem('stocksquad-auth-token', authToken);
    localStorage.setItem('stocksquad-user-id', email);
  };

  const logout = () => {
    setToken(null);
    setUserId(null);
    localStorage.removeItem('stocksquad-auth-token');
    localStorage.removeItem('stocksquad-user-id');
  };

  return (
    <AuthContext.Provider value={{ token, userId, login, logout, isAuthenticated: !!token, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
