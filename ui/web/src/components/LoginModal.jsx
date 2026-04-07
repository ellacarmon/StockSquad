import { useState } from 'react';
import { LogIn, User, Mail, Shield, Loader2, ArrowLeft } from 'lucide-react';
import * as api from '../utils/api.js';

export default function LoginModal({ onLogin }) {
  const [step, setStep] = useState(1); // 1 = email, 2 = code
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');

  const handleSendCode = async (e) => {
    e.preventDefault();

    // Basic email validation
    if (!email || !email.includes('@')) {
      setError('Please enter a valid email address');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const response = await api.post('/api/auth/send-code', { email });

      if (response.success) {
        setSuccess(response.message);
        setStep(2); // Move to code entry step
      } else {
        setError(response.message || 'Failed to send code');
      }
    } catch (err) {
      setError(err.message || 'Failed to send verification code. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyCode = async (e) => {
    e.preventDefault();

    if (!code || code.length !== 6) {
      setError('Please enter the 6-digit code');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const response = await api.post('/api/auth/verify-code', { email, code });

      if (response.success && response.token) {
        setSuccess(response.message);
        onLogin(response.token, response.email);
      } else {
        setError(response.message || 'Invalid verification code');
      }
    } catch (err) {
      if (err.status === 403) {
        setError('Your email is not authorized. Contact your administrator to get access.');
      } else if (err.status === 401) {
        setError('Invalid or expired code. Please try again.');
      } else {
        setError(err.message || 'Verification failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleBackToEmail = () => {
    setStep(1);
    setCode('');
    setError('');
    setSuccess('');
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        display: 'flex',
        alignments: 'center',
        justifyContent: 'center',
        zIndex: 10000,
        backdropFilter: 'blur(8px)',
      }}
    >
      <div
        style={{
          backgroundColor: 'var(--bg-card)',
          borderRadius: '20px',
          border: '1px solid var(--border-color)',
          boxShadow: '0 20px 80px rgba(0, 0, 0, 0.5)',
          padding: '48px',
          maxWidth: '480px',
          width: '90%',
          animation: 'fadeIn 0.3s ease-out',
        }}
      >
        {/* Step 1: Email */}
        {step === 1 && (
          <>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <div style={{
                width: '80px',
                height: '80px',
                borderRadius: '20px',
                background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 20px',
                boxShadow: '0 8px 24px rgba(59, 130, 246, 0.3)',
              }}>
                <Mail size={40} color="white" />
              </div>
              <h2 style={{
                fontSize: '2rem',
                fontWeight: 700,
                color: 'var(--text-main)',
                marginBottom: '8px',
              }}>
                Welcome to StockSquad
              </h2>
              <p style={{
                fontSize: '0.95rem',
                color: 'var(--text-muted)',
                lineHeight: 1.6,
              }}>
                Enter your email to receive a verification code
              </p>
            </div>

            <form onSubmit={handleSendCode}>
              <div style={{ marginBottom: '24px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  color: 'var(--text-main)',
                  marginBottom: '8px',
                }}>
                  Email Address
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    setError('');
                  }}
                  placeholder="your.email@example.com"
                  autoFocus
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    fontSize: '1rem',
                    backgroundColor: 'var(--bg-main)',
                    border: `2px solid ${error ? '#ef4444' : 'var(--border-color)'}`,
                    borderRadius: '12px',
                    color: 'var(--text-main)',
                    outline: 'none',
                    transition: 'all 0.2s',
                    boxSizing: 'border-box',
                  }}
                  onFocus={(e) => {
                    if (!error) e.target.style.borderColor = 'var(--accent-blue)';
                  }}
                  onBlur={(e) => {
                    if (!error) e.target.style.borderColor = 'var(--border-color)';
                  }}
                />
                {error && (
                  <p style={{
                    fontSize: '0.85rem',
                    color: '#ef4444',
                    marginTop: '8px',
                    marginBottom: 0,
                  }}>
                    {error}
                  </p>
                )}
              </div>

              <button
                type="submit"
                disabled={!email || loading}
                style={{
                  width: '100%',
                  padding: '14px 24px',
                  fontSize: '1rem',
                  fontWeight: 600,
                  backgroundColor: (email && !loading) ? '#3b82f6' : 'var(--bg-main)',
                  color: (email && !loading) ? 'white' : 'var(--text-muted)',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: (email && !loading) ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  transition: 'all 0.2s',
                  boxShadow: (email && !loading) ? '0 4px 16px rgba(59, 130, 246, 0.3)' : 'none',
                }}
              >
                {loading ? <Loader2 size={20} className="animate-spin" /> : <Mail size={20} />}
                {loading ? 'Sending...' : 'Send Code'}
              </button>
            </form>

            <div style={{
              marginTop: '24px',
              padding: '16px',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.2)',
              borderRadius: '12px',
            }}>
              <p style={{
                fontSize: '0.85rem',
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.5,
              }}>
                🔒 Only authorized emails can access. Contact your admin for access.
              </p>
            </div>
          </>
        )}

        {/* Step 2: Verification Code */}
        {step === 2 && (
          <>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <div style={{
                width: '80px',
                height: '80px',
                borderRadius: '20px',
                background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 20px',
                boxShadow: '0 8px 24px rgba(16, 185, 129, 0.3)',
              }}>
                <Shield size={40} color="white" />
              </div>
              <h2 style={{
                fontSize: '2rem',
                fontWeight: 700,
                color: 'var(--text-main)',
                marginBottom: '8px',
              }}>
                Check Your Email
              </h2>
              <p style={{
                fontSize: '0.95rem',
                color: 'var(--text-muted)',
                lineHeight: 1.6,
              }}>
                We sent a 6-digit code to <strong style={{color: 'var(--text-main)'}}>{email}</strong>
              </p>
            </div>

            {success && (
              <div style={{
                padding: '12px 16px',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '8px',
                marginBottom: '24px',
                color: '#10b981',
                fontSize: '0.9rem',
              }}>
                ✓ {success}
              </div>
            )}

            <form onSubmit={handleVerifyCode}>
              <div style={{ marginBottom: '24px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  color: 'var(--text-main)',
                  marginBottom: '8px',
                }}>
                  Verification Code
                </label>
                <input
                  type="text"
                  value={code}
                  onChange={(e) => {
                    const value = e.target.value.replace(/\D/g, '').slice(0, 6);
                    setCode(value);
                    setError('');
                  }}
                  placeholder="000000"
                  autoFocus
                  disabled={loading}
                  maxLength={6}
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    fontSize: '1.5rem',
                    fontWeight: 600,
                    textAlign: 'center',
                    letterSpacing: '0.5em',
                    backgroundColor: 'var(--bg-main)',
                    border: `2px solid ${error ? '#ef4444' : 'var(--border-color)'}`,
                    borderRadius: '12px',
                    color: 'var(--text-main)',
                    outline: 'none',
                    transition: 'all 0.2s',
                    boxSizing: 'border-box',
                    fontFamily: 'monospace',
                  }}
                  onFocus={(e) => {
                    if (!error) e.target.style.borderColor = 'var(--accent-blue)';
                  }}
                  onBlur={(e) => {
                    if (!error) e.target.style.borderColor = 'var(--border-color)';
                  }}
                />
                {error && (
                  <p style={{
                    fontSize: '0.85rem',
                    color: '#ef4444',
                    marginTop: '8px',
                    marginBottom: 0,
                    textAlign: 'center',
                  }}>
                    {error}
                  </p>
                )}
                <p style={{
                  fontSize: '0.8rem',
                  color: 'var(--text-muted)',
                  marginTop: '8px',
                  marginBottom: 0,
                  textAlign: 'center',
                }}>
                  Code expires in 10 minutes
                </p>
              </div>

              <button
                type="submit"
                disabled={code.length !== 6 || loading}
                style={{
                  width: '100%',
                  padding: '14px 24px',
                  fontSize: '1rem',
                  fontWeight: 600,
                  backgroundColor: (code.length === 6 && !loading) ? '#10b981' : 'var(--bg-main)',
                  color: (code.length === 6 && !loading) ? 'white' : 'var(--text-muted)',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: (code.length === 6 && !loading) ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  transition: 'all 0.2s',
                  boxShadow: (code.length === 6 && !loading) ? '0 4px 16px rgba(16, 185, 129, 0.3)' : 'none',
                  marginBottom: '12px',
                }}
              >
                {loading ? <Loader2 size={20} className="animate-spin" /> : <LogIn size={20} />}
                {loading ? 'Verifying...' : 'Verify & Sign In'}
              </button>

              <button
                type="button"
                onClick={handleBackToEmail}
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '12px 24px',
                  fontSize: '0.9rem',
                  fontWeight: 500,
                  backgroundColor: 'transparent',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '12px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  if (!loading) {
                    e.target.style.backgroundColor = 'var(--bg-hover)';
                    e.target.style.borderColor = 'var(--text-muted)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.target.style.backgroundColor = 'transparent';
                  e.target.style.borderColor = 'var(--border-color)';
                }}
              >
                <ArrowLeft size={16} />
                Back to Email
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
