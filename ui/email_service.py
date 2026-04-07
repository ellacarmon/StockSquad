"""
Email service for sending verification codes.
Supports multiple email providers: Resend, SendGrid, and SMTP.
"""

import os
import random
import string
from typing import Optional
from datetime import datetime, timedelta


class EmailService:
    """Service for sending verification emails."""

    def __init__(self):
        """Initialize email service with configured provider."""
        self.provider = os.getenv("EMAIL_PROVIDER", "resend").lower()

        if self.provider == "resend":
            self._init_resend()
        elif self.provider == "sendgrid":
            self._init_sendgrid()
        elif self.provider == "smtp":
            self._init_smtp()
        elif self.provider == "acs" or self.provider == "azure":
            self._init_acs()
        else:
            print(f"WARNING: Unknown email provider '{self.provider}'. Email sending disabled.")
            self.enabled = False

    def _init_resend(self):
        """Initialize Resend email service."""
        api_key = os.getenv("RESEND_API_KEY", "")
        if not api_key:
            print("WARNING: RESEND_API_KEY not set. Email sending disabled.")
            self.enabled = False
            return

        try:
            import resend
            resend.api_key = api_key
            self.resend = resend
            self.enabled = True
            print("Email service enabled: Resend")
        except ImportError:
            print("ERROR: resend package not installed. Run: pip install resend")
            self.enabled = False

    def _init_sendgrid(self):
        """Initialize SendGrid email service."""
        api_key = os.getenv("SENDGRID_API_KEY", "")
        if not api_key:
            print("WARNING: SENDGRID_API_KEY not set. Email sending disabled.")
            self.enabled = False
            return

        try:
            from sendgrid import SendGridAPIClient
            self.sendgrid_client = SendGridAPIClient(api_key)
            self.enabled = True
            print("Email service enabled: SendGrid")
        except ImportError:
            print("ERROR: sendgrid package not installed. Run: pip install sendgrid")
            self.enabled = False

    def _init_smtp(self):
        """Initialize SMTP email service."""
        self.smtp_host = os.getenv("SMTP_HOST", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

        if not all([self.smtp_host, self.smtp_user, self.smtp_password]):
            print("WARNING: SMTP credentials not set. Email sending disabled.")
            self.enabled = False
            return

        self.enabled = True
        print(f"Email service enabled: SMTP ({self.smtp_host})")

    def _init_acs(self):
        """Initialize Azure Communication Services email."""
        connection_string = os.getenv("ACS_CONNECTION_STRING", "")
        if not connection_string:
            print("WARNING: ACS_CONNECTION_STRING not set. Email sending disabled.")
            self.enabled = False
            return

        try:
            from azure.communication.email import EmailClient
            self.acs_client = EmailClient.from_connection_string(connection_string)
            self.enabled = True
            print("✅ Email service enabled: Azure Communication Services")
        except ImportError:
            print("ERROR: azure-communication-email package not installed. Run: pip install azure-communication-email")
            self.enabled = False
        except Exception as e:
            print(f"ERROR: Failed to initialize Azure Communication Services: {e}")
            self.enabled = False

    async def send_verification_code(self, to_email: str, code: str) -> bool:
        """
        Send verification code to email.

        Args:
            to_email: Recipient email address
            code: 6-digit verification code

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            print(f"Email service not enabled. Would send code {code} to {to_email}")
            # In development, print the code so you can test
            print(f"🔑 VERIFICATION CODE FOR {to_email}: {code}")
            return True  # Return True in dev mode so testing works

        try:
            if self.provider == "resend":
                return await self._send_resend(to_email, code)
            elif self.provider == "sendgrid":
                return await self._send_sendgrid(to_email, code)
            elif self.provider == "smtp":
                return await self._send_smtp(to_email, code)
            elif self.provider == "acs" or self.provider == "azure":
                return await self._send_acs(to_email, code)
        except Exception as e:
            print(f"Failed to send email to {to_email}: {e}")
            return False

        return False

    async def _send_resend(self, to_email: str, code: str) -> bool:
        """Send email via Resend."""
        from_email = os.getenv("EMAIL_FROM", "noreply@stocksquad.app")

        params = {
            "from": from_email,
            "to": [to_email],
            "subject": "Your StockSquad Verification Code",
            "html": f"""
            <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 28px;">StockSquad</h1>
                    </div>
                    <div style="background: #f8fafc; padding: 40px; border-radius: 0 0 10px 10px;">
                        <h2 style="color: #1e293b; margin-top: 0;">Your Verification Code</h2>
                        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
                            Enter this code to complete your login:
                        </p>
                        <div style="background: white; border: 2px solid #3b82f6; border-radius: 10px; padding: 20px; text-align: center; margin: 30px 0;">
                            <span style="font-size: 36px; font-weight: bold; color: #3b82f6; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                                {code}
                            </span>
                        </div>
                        <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                            This code will expire in <strong>10 minutes</strong>.
                        </p>
                        <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                            If you didn't request this code, you can safely ignore this email.
                        </p>
                    </div>
                    <div style="text-align: center; margin-top: 20px; color: #94a3b8; font-size: 12px;">
                        <p>StockSquad - AI-Powered Stock Analysis</p>
                    </div>
                </body>
            </html>
            """
        }

        try:
            self.resend.Emails.send(params)
            print(f"✅ Verification code sent to {to_email}")
            return True
        except Exception as e:
            print(f"❌ Resend error: {e}")
            return False

    async def _send_sendgrid(self, to_email: str, code: str) -> bool:
        """Send email via SendGrid."""
        from sendgrid.helpers.mail import Mail, Email, To, Content

        from_email = os.getenv("EMAIL_FROM", "noreply@stocksquad.app")

        message = Mail(
            from_email=Email(from_email),
            to_emails=To(to_email),
            subject="Your StockSquad Verification Code",
            html_content=Content("text/html", f"""
            <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 28px;">StockSquad</h1>
                    </div>
                    <div style="background: #f8fafc; padding: 40px; border-radius: 0 0 10px 10px;">
                        <h2 style="color: #1e293b; margin-top: 0;">Your Verification Code</h2>
                        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
                            Enter this code to complete your login:
                        </p>
                        <div style="background: white; border: 2px solid #3b82f6; border-radius: 10px; padding: 20px; text-align: center; margin: 30px 0;">
                            <span style="font-size: 36px; font-weight: bold; color: #3b82f6; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                                {code}
                            </span>
                        </div>
                        <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                            This code will expire in <strong>10 minutes</strong>.
                        </p>
                    </div>
                </body>
            </html>
            """)
        )

        try:
            response = self.sendgrid_client.send(message)
            print(f"✅ Verification code sent to {to_email}")
            return True
        except Exception as e:
            print(f"❌ SendGrid error: {e}")
            return False

    async def _send_smtp(self, to_email: str, code: str) -> bool:
        """Send email via SMTP."""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        from_email = os.getenv("EMAIL_FROM", self.smtp_user)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your StockSquad Verification Code"
        msg["From"] = from_email
        msg["To"] = to_email

        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 28px;">StockSquad</h1>
                </div>
                <div style="background: #f8fafc; padding: 40px; border-radius: 0 0 10px 10px;">
                    <h2 style="color: #1e293b; margin-top: 0;">Your Verification Code</h2>
                    <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
                        Enter this code to complete your login:
                    </p>
                    <div style="background: white; border: 2px solid #3b82f6; border-radius: 10px; padding: 20px; text-align: center; margin: 30px 0;">
                        <span style="font-size: 36px; font-weight: bold; color: #3b82f6; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                            {code}
                        </span>
                    </div>
                    <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                        This code will expire in <strong>10 minutes</strong>.
                    </p>
                </div>
            </body>
        </html>
        """

        part = MIMEText(html, "html")
        msg.attach(part)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(from_email, to_email, msg.as_string())

            print(f"✅ Verification code sent to {to_email}")
            return True
        except Exception as e:
            print(f"❌ SMTP error: {e}")
            return False

    async def _send_acs(self, to_email: str, code: str) -> bool:
        """Send email via Azure Communication Services."""
        from_email = os.getenv("EMAIL_FROM", "DoNotReply@stocksquad.app")

        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 28px;">StockSquad</h1>
                </div>
                <div style="background: #f8fafc; padding: 40px; border-radius: 0 0 10px 10px;">
                    <h2 style="color: #1e293b; margin-top: 0;">Your Verification Code</h2>
                    <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
                        Enter this code to complete your login:
                    </p>
                    <div style="background: white; border: 2px solid #3b82f6; border-radius: 10px; padding: 20px; text-align: center; margin: 30px 0;">
                        <span style="font-size: 36px; font-weight: bold; color: #3b82f6; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                            {code}
                        </span>
                    </div>
                    <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                        This code will expire in <strong>10 minutes</strong>.
                    </p>
                    <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                        If you didn't request this code, you can safely ignore this email.
                    </p>
                </div>
                <div style="text-align: center; margin-top: 20px; color: #94a3b8; font-size: 12px;">
                    <p>StockSquad - AI-Powered Stock Analysis</p>
                </div>
            </body>
        </html>
        """

        message = {
            "senderAddress": from_email,
            "recipients": {
                "to": [{"address": to_email}]
            },
            "content": {
                "subject": "Your StockSquad Verification Code",
                "html": html_content
            }
        }

        try:
            poller = self.acs_client.begin_send(message)
            result = poller.result()
            print(f"✅ Verification code sent to {to_email} via Azure Communication Services")
            print(f"   Message ID: {result['id']}")
            return True
        except Exception as e:
            print(f"❌ Azure Communication Services error: {e}")
            return False


# In-memory storage for verification codes (use Redis in production)
verification_codes = {}

def generate_verification_code() -> str:
    """Generate a 6-digit verification code."""
    return ''.join(random.choices(string.digits, k=6))

def store_verification_code(email: str, code: str, expires_in_minutes: int = 10):
    """Store verification code with expiration."""
    expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
    verification_codes[email.lower()] = {
        "code": code,
        "expires_at": expires_at,
        "attempts": 0
    }
    print(f"📝 Stored verification code for {email} (expires in {expires_in_minutes} min)")

def verify_code(email: str, code: str) -> bool:
    """Verify the code matches and hasn't expired."""
    email = email.lower()

    if email not in verification_codes:
        print(f"❌ No verification code found for {email}")
        return False

    stored = verification_codes[email]

    # Check expiration
    if datetime.utcnow() > stored["expires_at"]:
        print(f"❌ Verification code expired for {email}")
        del verification_codes[email]
        return False

    # Check attempts (max 3)
    if stored["attempts"] >= 3:
        print(f"❌ Too many attempts for {email}")
        del verification_codes[email]
        return False

    # Check code
    if stored["code"] != code:
        stored["attempts"] += 1
        print(f"❌ Invalid code for {email} (attempt {stored['attempts']}/3)")
        return False

    # Success! Clean up
    del verification_codes[email]
    print(f"✅ Verification successful for {email}")
    return True


# Global email service instance
email_service = EmailService()
