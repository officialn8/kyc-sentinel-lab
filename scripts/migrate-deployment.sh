#!/bin/bash
# Migration script for updating existing deployments
# This helps transition from old configuration to new security requirements

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🔄 KYC Sentinel Lab - Deployment Migration Assistant"
echo "==================================================="
echo
echo "This script helps migrate your existing deployment to the new security requirements."
echo

# Check if we're in production/staging
read -p "Which environment are you migrating? (production/staging): " ENV_TYPE

if [ "$ENV_TYPE" != "production" ] && [ "$ENV_TYPE" != "staging" ]; then
    echo -e "${RED}Invalid environment. Must be 'production' or 'staging'${NC}"
    exit 1
fi

echo
echo -e "${YELLOW}Step 1: Generate Required Secrets${NC}"
echo "-----------------------------------"

# Check for existing API key
if [ -z "${BACKEND_API_KEY:-}" ]; then
    echo -e "${BLUE}Generating new API key...${NC}"
    NEW_API_KEY=$(openssl rand -base64 32)
    echo -e "${GREEN}Generated API Key:${NC} $NEW_API_KEY"
    echo
    echo -e "${YELLOW}⚠️  Save this API key securely! You'll need to add it to:${NC}"
    echo "   - Railway: BACKEND_API_KEY environment variable"
    echo "   - Any API clients: X-API-Key header"
    echo
    read -p "Press enter when you've saved the API key..."
else
    echo -e "${GREEN}✓ Using existing BACKEND_API_KEY${NC}"
fi

# Generate webhook secret
echo
echo -e "${BLUE}Generating webhook secret...${NC}"
NEW_WEBHOOK_SECRET=$(openssl rand -base64 32)
echo -e "${GREEN}Generated Webhook Secret:${NC} $NEW_WEBHOOK_SECRET"
echo
echo -e "${YELLOW}⚠️  Save this webhook secret! Add it to Railway as WEBHOOK_SECRET${NC}"
echo
read -p "Press enter when you've saved the webhook secret..."

echo
echo -e "${YELLOW}Step 2: Update Railway Environment Variables${NC}"
echo "--------------------------------------------"
echo
echo "Add or update these variables in Railway:"
echo
echo -e "${GREEN}Required variables:${NC}"
echo "  ENVIRONMENT=$ENV_TYPE"
echo "  BACKEND_API_KEY=<your-api-key>"
echo "  WEBHOOK_SECRET=<your-webhook-secret>"
echo "  DEBUG=false"
echo "  AUTH_DISABLED=false  (or remove it entirely)"
echo
echo -e "${YELLOW}If you have AUTH_DISABLED=true, remove it!${NC}"
echo
read -p "Press enter when you've updated Railway variables..."

echo
echo -e "${YELLOW}Step 3: Update Frontend Code${NC}"
echo "----------------------------"
echo
echo "The API response format has changed. Update your upload code:"
echo
echo -e "${RED}Old code:${NC}"
echo "  const response = await api.createSession({...});"
echo "  await uploadToPresignedUrl(response.upload_urls.selfie_upload_url, file);"
echo
echo -e "${GREEN}New code:${NC}"
echo "  const response = await api.createSession({...});"
echo "  await uploadToPresignedPost(response.selfie_upload, file, onProgress);"
echo
read -p "Have you updated the frontend code? (y/n): " FRONTEND_UPDATED

if [ "$FRONTEND_UPDATED" != "y" ]; then
    echo -e "${RED}⚠️  Frontend must be updated before the backend deployment!${NC}"
    echo "Update the code and redeploy frontend first."
    exit 1
fi

echo
echo -e "${YELLOW}Step 4: Deployment Order${NC}"
echo "------------------------"
echo
echo -e "${GREEN}Deploy in this order to avoid downtime:${NC}"
echo "1. Deploy frontend with new upload code (supports both formats)"
echo "2. Deploy backend with new security requirements"
echo "3. Verify everything works"
echo "4. Remove deprecated uploadToPresignedUrl from frontend (cleanup)"
echo
echo -e "${BLUE}Ready to deploy?${NC}"
echo
echo "Backend deployment checklist:"
echo "  [ ] ENVIRONMENT=$ENV_TYPE is set"
echo "  [ ] BACKEND_API_KEY is set"
echo "  [ ] WEBHOOK_SECRET is set"
echo "  [ ] DEBUG=false"
echo "  [ ] AUTH_DISABLED is removed or false"
echo "  [ ] Frontend is already deployed with new code"
echo
read -p "Confirm all items are checked (y/n): " CONFIRM

if [ "$CONFIRM" = "y" ]; then
    echo
    echo -e "${GREEN}✅ Ready to deploy!${NC}"
    echo
    echo "Deploy commands:"
    echo "  Railway: Push to your connected branch or 'railway up'"
    echo "  Manual: docker build & push, then update service"
    echo
    echo "After deployment:"
    echo "  1. Check health: curl https://your-api/health"
    echo "  2. Test upload with new format"
    echo "  3. Monitor logs for any auth errors"
else
    echo
    echo -e "${YELLOW}Complete the checklist before deploying.${NC}"
fi

echo
echo -e "${BLUE}Need help?${NC}"
echo "- Check logs: railway logs"
echo "- Validate config: ./scripts/deploy-check.sh"
echo "- See full guide: deploy/DEPLOYMENT_GUIDE.md"