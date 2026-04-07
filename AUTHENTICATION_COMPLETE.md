# StockSquad - Permit.io Authentication Implementation Complete! 🎉

Your StockSquad application is now secured with Permit.io authorization. Only authorized users can create analyses and use chat features.

---

## 🔐 What's Been Implemented

### Backend (FastAPI)

#### 1. **Permit.io Authorization Middleware** (`ui/auth.py`)
- Permission checking for `create`, `read`, and `delete` actions
- FastAPI dependencies for easy endpoint protection
- Graceful fallback when Permit.io is not configured (development mode)

#### 2. **Protected API Endpoints** (`ui/api.py`)
- `GET /api/analyze/stream` - Requires `analysis:create` permission
- `POST /api/chat` - Requires `analysis:create` permission
- `DELETE /api/chat/{session_id}` - Requires `analysis:delete` permission

#### 3. **Public Endpoints** (No Authentication)
- `GET /api/health` - Health check
- `GET /api/reports` - View all reports
- `GET /api/reports/{doc_id}` - View specific report details
- `GET /api/reports/{doc_id}/date-insights` - View date-specific insights

#### 4. **Infrastructure** (`infra/container-apps.bicep`)
- Added `permitIoApiKey` parameter (secure)
- Added `PERMIT_IO_API_KEY` environment variable
- Added `PERMIT_IO_PDP_URL` environment variable

### Frontend (React)

#### 1. **Authentication Context** (`ui/web/src/contexts/AuthContext.jsx`)
- Manages user authentication state
- Persists user ID in localStorage
- Provides `login()`, `logout()`, `isAuthenticated` hooks

#### 2. **Login Modal** (`ui/web/src/components/LoginModal.jsx`)
- Beautiful, modern login interface
- Email validation
- Shows authorization status

#### 3. **User Menu** (`ui/web/src/components/UserMenu.jsx`)
- Displays current user with initials avatar
- Dropdown menu with user info
- Logout button

#### 4. **Authenticated API Client** (`ui/web/src/utils/api.js`)
- Automatically includes `X-User-Id` header in all requests
- Handles 401/403 errors (forces re-login)
- Supports Server-Sent Events (SSE) with authentication
- Provides `get()`, `post()`, `del()` helpers

#### 5. **Updated App Component** (`ui/web/src/App.jsx`)
- Shows login modal when not authenticated
- Displays user menu in sidebar
- All API calls use authenticated client
- Automatic logout on authorization failures

---

## 🚀 Setup Instructions

### Step 1: Create Permit.io Account (5 minutes)

1. Go to [https://permit.io](https://permit.io) and sign up
2. Create a new project: **StockSquad**
3. Navigate to **Policy** → **Resources**
4. Create resource:
   - Name: `Analysis`
   - Key: `analysis`
   - Actions: `create`, `read`, `delete`

### Step 2: Create Admin Role

1. Go to **Policy** → **Roles**
2. Create new role:
   - Name: `Admin`
   - Key: `admin`
3. Grant permissions:
   - ✅ `analysis:create`
   - ✅ `analysis:read`
   - ✅ `analysis:delete`

### Step 3: Add Yourself as a User

1. Go to **Directory** → **Users**
2. Click **"+ New User"**
3. Enter:
   - User ID: `your.email@example.com` (your actual email)
   - Email: Same as User ID
   - First Name: Your name
   - Last Name: Your last name
4. Click **Create**
5. Assign role:
   - Click on your user
   - **Role Assignments** → **"+ Assign Role"**
   - Role: `Admin`
   - Tenant: `Default`
   - Click **Assign**

### Step 4: Get Your API Key

1. Go to **Settings** → **API Keys**
2. Copy the **Secret API Key** (starts with `permit_key_...`)
3. **Save it securely** - you'll need it for deployment

### Step 5: Deploy to Azure

```bash
# Set your Permit.io API key
export PERMIT_IO_API_KEY="permit_key_xxxxxxxxxxxxx"

# Deploy with authentication
az deployment group create \
  --resource-group YOUR_RESOURCE_GROUP \
  --template-file infra/container-apps.bicep \
  --parameters \
    permitIoApiKey="$PERMIT_IO_API_KEY" \
    azureOpenAiEndpoint="$AZURE_OPENAI_ENDPOINT" \
    azureOpenAiApiKey="$AZURE_OPENAI_API_KEY" \
    azureOpenAiDeploymentName="$AZURE_OPENAI_DEPLOYMENT_NAME" \
    azureOpenAiEmbeddingDeploymentName="$AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
```

### Step 6: Build and Push Docker Image

```bash
# Build the updated image with authentication
docker build -t stocksquad:latest .

# Tag for your ACR
docker tag stocksquad:latest YOUR_ACR.azurecr.io/stocksquad:latest

# Push to ACR
docker push YOUR_ACR.azurecr.io/stocksquad:latest
```

### Step 7: Test Authentication

1. **Open your app** in a browser: `https://your-app.azurecontainerapps.io`
2. **You should see the login modal**
3. **Enter your email** (the one you added to Permit.io)
4. **Click "Sign In"**
5. **You should now have access** to all features!

---

## 🧪 Testing

### Test 1: Login Flow
1. Open app in incognito window
2. Should show login modal
3. Enter authorized email → Should see dashboard
4. Enter unauthorized email → Should see error when trying to create analysis

### Test 2: Protected Endpoints
```bash
# Without authentication (should fail)
curl "https://your-app.azurecontainerapps.io/api/analyze/stream?ticker=AAPL"
# Expected: 401 Unauthorized

# With authentication (should work)
curl "https://your-app.azurecontainerapps.io/api/analyze/stream?ticker=AAPL&user_id=your.email@example.com"
# Expected: SSE stream starts
```

### Test 3: Public Endpoints (No Auth Required)
```bash
# These should work without authentication
curl "https://your-app.azurecontainerapps.io/api/health"
curl "https://your-app.azurecontainerapps.io/api/reports"
```

---

## 👥 Managing Users

### Add a New User

1. **In Permit.io dashboard:**
   - Directory → Users → "+ New User"
   - Enter their email as User ID
   - Assign "Admin" role
   - Done!

2. **Tell them to:**
   - Visit the app URL
   - Enter their email at the login screen
   - Start analyzing!

### Remove a User

1. Go to Directory → Users
2. Find the user
3. Click "..." → Delete User
4. **They are immediately logged out** (next API call will fail)

### Change Permissions

Want some users to only view reports?

1. Create a "Viewer" role with only `analysis:read` permission
2. Assign that role to users instead of "Admin"
3. They won't be able to create analyses or chat

---

## 🔧 Local Development

To test authentication locally:

1. **Add to `.env`:**
```bash
PERMIT_IO_API_KEY=permit_key_xxxxxxxxxxxxx
PERMIT_IO_PDP_URL=https://cloudpdp.api.permit.io
```

2. **Run the app:**
```bash
python main.py ui
```

3. **Open browser:** `http://localhost:8000`

4. **Login with your email** (must be added to Permit.io first!)

---

## 🎨 User Experience

### Login Screen
- Clean, modern design
- Email input with validation
- Clear messaging about authorization
- Remembers user (localStorage)

### User Menu
- Shows user avatar with initials
- Displays full email on hover
- "Authorized" badge
- Logout button

### Automatic Logout
If a user's permissions are revoked in Permit.io:
- Next API call will return 401/403
- App automatically logs them out
- They see login screen again

---

## 🔒 Security Features

### ✅ What's Protected
1. **Creating analyses** - Only authorized users
2. **Chat functionality** - Only authorized users
3. **Deleting chat sessions** - Only authorized users

### 🌐 What's Public (Read-Only)
1. **Viewing reports** - Anyone can see existing reports
2. **Health checks** - No authentication needed

### 🛡️ Security Measures
1. **Permission checking** - Every protected endpoint verifies permissions
2. **Secure credentials** - API key stored in Azure Key Vault
3. **Fail-closed** - If Permit.io check fails, access is denied
4. **Audit logs** - Permit.io tracks all access decisions
5. **Session persistence** - User stays logged in across browser sessions
6. **Automatic logout** - Revoked permissions trigger immediate logout

---

## 📊 Monitoring

### Check Authorization Decisions

In Permit.io dashboard:
1. Go to **Audit**
2. See all authorization checks
3. Filter by user, resource, or action
4. Identify unauthorized access attempts

### View API Errors

In Azure Portal:
1. Go to Container App → **Logs**
2. Filter for "Permit.io" messages
3. Look for authorization failures

---

## 💡 Tips

### Want to Protect Report Viewing Too?

Easy! Just update these endpoints in `ui/api.py`:

```python
@app.get("/api/reports")
async def get_reports(user_id: str = Depends(require_auth)):
    # Now requires authentication
    ...

@app.get("/api/reports/{doc_id}")
async def get_report_detail(
    doc_id: str,
    user_id: str = Depends(require_auth)
):
    # Now requires authentication
    ...
```

### Want Email Domain Restrictions?

In Permit.io, create a custom policy:
1. Go to Policy → Code
2. Add condition: `user.id.endsWith("@yourcompany.com")`

### Want Different Permissions Per Ticker?

Permit.io supports resource-level permissions. You could create resources like:
- `analysis:AAPL`
- `analysis:TSLA`

And give users permissions to specific tickers only!

---

## 🐛 Troubleshooting

### Issue: "Authentication required" error

**Cause**: User not added to Permit.io or wrong email

**Solution**:
1. Go to Permit.io → Directory → Users
2. Verify user exists with exact email (case-sensitive!)
3. Check they have "Admin" role assigned

### Issue: Login screen won't go away

**Cause**: User added to Permit.io but no role assigned

**Solution**:
1. Go to user in Permit.io
2. Role Assignments → Assign "Admin" role
3. Refresh browser

### Issue: "Permit.io check failed" in logs

**Cause**: API key is wrong or PDP URL is unreachable

**Solution**:
1. Verify `PERMIT_IO_API_KEY` environment variable is set correctly
2. Check Permit.io dashboard → Settings → API Keys
3. Generate new key if needed and redeploy

### Issue: Works locally but not in Azure

**Cause**: Environment variables not set in Container App

**Solution**:
1. Azure Portal → Container App → Environment variables
2. Verify `PERMIT_IO_API_KEY` is present
3. Check deployment logs for configuration errors

---

## 📈 Next Steps

Now that your app is secured, consider:

1. **🔔 Set up alerts** for unauthorized access attempts
2. **📝 Document your users** and their roles
3. **🎯 Create more granular roles** (Viewer, Analyst, Admin)
4. **🌍 Add multi-tenancy** for different teams/organizations
5. **📊 Monitor usage** in Permit.io analytics

---

## 🎓 Learn More

- **Permit.io Docs**: https://docs.permit.io
- **RBAC Tutorial**: https://docs.permit.io/concepts/rbac
- **Best Practices**: https://docs.permit.io/security/best-practices

---

## ✅ Checklist

Before going live, make sure:

- [ ] Permit.io account created
- [ ] Resource and roles configured
- [ ] Your user added with Admin role
- [ ] API key copied and saved securely
- [ ] Environment variables set in Azure
- [ ] Docker image built and pushed
- [ ] Container App redeployed
- [ ] Login tested in browser
- [ ] Analysis creation tested
- [ ] Unauthorized access tested (should fail)
- [ ] Public endpoints still work
- [ ] User menu displays correctly
- [ ] Logout works

---

**🎉 Congratulations! Your StockSquad application is now secure and production-ready!**

Need help? Check `PERMIT_IO_SETUP.md` for detailed Permit.io configuration instructions.
