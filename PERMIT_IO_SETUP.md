# StockSquad - Permit.io Authorization Setup

This guide will help you set up Permit.io for user management and authorization in your StockSquad application.

## Why Permit.io?

Permit.io provides:
- **Easy User Management**: Add/remove users via dashboard (no code changes needed)
- **Role-Based Access Control (RBAC)**: Assign roles and permissions
- **Audit Logs**: Track who did what and when
- **Policy Management**: Control access with fine-grained policies
- **Free Tier**: Up to 1,000 monthly active users

---

## Step 1: Create a Permit.io Account

1. Go to [https://permit.io](https://permit.io)
2. Click **"Get Started Free"**
3. Sign up with your email or GitHub account
4. Verify your email

---

## Step 2: Create Your First Project

1. After logging in, you'll be prompted to create a project
2. Name it: **StockSquad**
3. Click **Create Project**

---

## Step 3: Configure Resources and Actions

### 3.1 Create the "Analysis" Resource

1. In the Permit.io dashboard, go to **Policy** → **Resources**
2. Click **"+ New Resource"**
3. Fill in:
   - **Name**: `Analysis`
   - **Key**: `analysis` (auto-generated)
   - **Description**: `Stock analysis operations`
4. Click **Create**

### 3.2 Add Actions to the Resource

1. Click on the **Analysis** resource you just created
2. Add these actions:
   - `create` - Create new stock analyses
   - `read` - View existing analyses
   - `delete` - Delete analyses and chat sessions
3. Click **Save**

---

## Step 4: Create Roles

### 4.1 Create "Admin" Role

1. Go to **Policy** → **Roles**
2. Click **"+ New Role"**
3. Fill in:
   - **Name**: `Admin`
   - **Key**: `admin`
   - **Description**: `Full access to all operations`
4. Under **Permissions**, grant:
   - ✅ `analysis:create`
   - ✅ `analysis:read`
   - ✅ `analysis:delete`
5. Click **Create**

### 4.2 Create "Viewer" Role (Optional)

If you want some users to only view reports:

1. Click **"+ New Role"**
2. Fill in:
   - **Name**: `Viewer`
   - **Key**: `viewer`
   - **Description**: `Read-only access`
3. Under **Permissions**, grant:
   - ✅ `analysis:read`
4. Click **Create**

---

## Step 5: Add Your User

1. Go to **Directory** → **Users**
2. Click **"+ New User"**
3. Fill in:
   - **User ID**: Your email address (e.g., `ella@example.com`)
   - **Email**: Same as User ID
   - **First Name**: Your first name
   - **Last Name**: Your last name
4. Click **Create**

### 5.1 Assign Role to Your User

1. Click on the user you just created
2. Go to **Role Assignments**
3. Click **"+ Assign Role"**
4. Select:
   - **Role**: `Admin`
   - **Tenant**: `Default`
5. Click **Assign**

---

## Step 6: Get Your API Key

1. Go to **Settings** → **API Keys**
2. Copy the **Secret API Key** (starts with `permit_key_...`)
3. **IMPORTANT**: Save this key securely - you'll need it for deployment

---

## Step 7: Deploy with Permit.io

### Option A: Deploy to Azure Container Apps

```bash
# Set your Permit.io API key
PERMIT_IO_API_KEY="permit_key_xxxxxxxxxxxxxxxxx"

# Deploy with Permit.io enabled
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

### Option B: Local Development

Add to your `.env` file:

```bash
PERMIT_IO_API_KEY=permit_key_xxxxxxxxxxxxxxxxx
PERMIT_IO_PDP_URL=https://cloudpdp.api.permit.io
```

Then run:
```bash
python main.py ui
```

---

## Step 8: Test Authentication

### 8.1 Test Without Authentication (Should Fail)

```bash
# This should return 401 Unauthorized
curl -X GET "https://your-app-url.azurecontainerapps.io/api/analyze/stream?ticker=AAPL"
```

Expected response:
```json
{
  "detail": "Authentication required. Please provide X-User-Id header."
}
```

### 8.2 Test With Authentication (Should Work)

```bash
# This should work
curl -X GET "https://your-app-url.azurecontainerapps.io/api/analyze/stream?ticker=AAPL" \
  -H "X-User-Id: ella@example.com"
```

### 8.3 Test With Unauthorized User (Should Fail)

```bash
# Add a user in Permit.io without assigning a role
# Then try to access with that user
curl -X GET "https://your-app-url.azurecontainerapps.io/api/analyze/stream?ticker=AAPL" \
  -H "X-User-Id: unauthorized@example.com"
```

Expected response:
```json
{
  "detail": "User 'unauthorized@example.com' is not authorized to create analysis."
}
```

---

## Managing Users

### Add a New User

1. Go to Permit.io dashboard → **Directory** → **Users**
2. Click **"+ New User"**
3. Enter their email as User ID
4. Assign them the **Admin** role (or **Viewer** for read-only)
5. Done! They can now access the app

### Remove a User

1. Go to **Directory** → **Users**
2. Find the user
3. Click **"..."** → **Delete User**
4. Confirm deletion

### Change User Permissions

1. Go to **Directory** → **Users**
2. Click on the user
3. Go to **Role Assignments**
4. Add or remove role assignments

---

## Using the Web UI with Authentication

The web UI needs to send the `X-User-Id` header with every request. You have two options:

### Option 1: Update React App to Prompt for User ID

The React app should:
1. Prompt for user email on first load
2. Store it in localStorage
3. Add `X-User-Id` header to all API requests

### Option 2: Simple Authentication Page (I can implement this)

Add a login page where users enter their email, then the app includes it in all API calls.

Would you like me to implement this in your React frontend?

---

## API Endpoints and Required Permissions

### Protected Endpoints (Require Authentication)

| Endpoint | Method | Required Permission | Description |
|----------|--------|---------------------|-------------|
| `/api/analyze/stream` | GET | `analysis:create` | Create new stock analysis |
| `/api/chat` | POST | `analysis:create` | Chat about analyses |
| `/api/chat/{session_id}` | DELETE | `analysis:delete` | Delete chat session |

### Public Endpoints (No Authentication Required)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/reports` | GET | List all reports |
| `/api/reports/{doc_id}` | GET | Get specific report |
| `/api/reports/{doc_id}/date-insights` | GET | Get date-specific insights |
| `/api/chat/{session_id}/history` | GET | Get chat history |

**Note**: If you want to protect the report viewing endpoints, I can add that too!

---

## Advanced Configuration

### Enable Email Restrictions

If you only want specific email domains (e.g., only `@yourcompany.com`):

1. In Permit.io, create a custom policy
2. Add a condition to check user email domain
3. The free tier supports basic policies

### Multi-Tenancy

If you want to isolate data between different teams/organizations:

1. Create **Tenants** in Permit.io
2. Assign users to specific tenants
3. Policies can be scoped to tenants

### Audit Logs

View all authorization decisions:

1. Go to **Audit** in Permit.io dashboard
2. See who accessed what and when
3. Filter by user, resource, or action

---

## Troubleshooting

### Error: "PERMIT_IO_API_KEY not set"

**Solution**: Make sure you've set the environment variable in your deployment:
- For Azure Container Apps: Pass `permitIoApiKey` parameter
- For local dev: Add to `.env` file

### Error: "User not authorized"

**Possible causes**:
1. User doesn't exist in Permit.io → Add them in the dashboard
2. User doesn't have the right role → Assign the "Admin" role
3. Wrong User ID in header → Make sure `X-User-Id` matches the User ID in Permit.io (case-sensitive!)

### Permit.io Check Fails

If you see "Permit.io check failed" in logs:
1. Check that your API key is correct
2. Verify PDP URL is accessible: `https://cloudpdp.api.permit.io`
3. Check Permit.io dashboard for service status

### Development Mode (No Authentication)

If `PERMIT_IO_API_KEY` is not set, the app runs in **development mode** with authorization disabled. This is useful for local testing but **NEVER deploy to production without setting the API key**.

---

## Cost

**Permit.io Free Tier:**
- Up to 1,000 monthly active users
- Unlimited policy checks
- Full RBAC features
- Community support

**Paid Tiers:**
- Start at $199/month for 5,000 MAU
- Advanced features like ABAC, custom roles
- Priority support

For a personal StockSquad app, the free tier is more than enough!

---

## Security Best Practices

1. **Never commit API keys** to git - use environment variables
2. **Use HTTPS** for all API calls (Container Apps uses HTTPS by default)
3. **Rotate API keys** periodically in Permit.io dashboard
4. **Monitor audit logs** to detect suspicious activity
5. **Use specific email addresses** as User IDs (not generic IDs like "user1")

---

## Next Steps

After setting up Permit.io:

1. ✅ Test authentication with your user
2. ✅ Add additional users if needed
3. 🔲 Update React frontend to send `X-User-Id` header (let me know if you need help!)
4. 🔲 Set up monitoring/alerts for authorization failures
5. 🔲 Document your users and their roles

---

## Questions?

- **Permit.io Docs**: https://docs.permit.io
- **Support**: support@permit.io
- **Community**: https://permit.io/community

Need help with setup? Let me know!
