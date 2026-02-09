#!/bin/bash
# Deployment environment validation script
# This script validates that all required environment variables are set
# before deploying to production or staging environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 KYC Sentinel Lab - Deployment Environment Check"
echo "=================================================="

# Get environment type
ENVIRONMENT=${ENVIRONMENT:-"not_set"}

echo -e "\nEnvironment: ${YELLOW}${ENVIRONMENT}${NC}"

# Function to check required variable
check_required() {
    local var_name=$1
    local var_value=${!var_name:-""}

    if [ -z "$var_value" ]; then
        echo -e "${RED}❌ Missing: ${var_name}${NC}"
        return 1
    else
        # Mask sensitive values
        if [[ $var_name == *"KEY"* ]] || [[ $var_name == *"SECRET"* ]] || [[ $var_name == *"PASSWORD"* ]]; then
            echo -e "${GREEN}✅ Set: ${var_name}${NC} (***masked***)"
        else
            echo -e "${GREEN}✅ Set: ${var_name}${NC} = ${var_value}"
        fi
        return 0
    fi
}

# Function to check forbidden variable
check_forbidden() {
    local var_name=$1
    local var_value=${!var_name:-""}

    if [ "$var_value" = "true" ]; then
        echo -e "${RED}❌ Forbidden: ${var_name}=true in ${ENVIRONMENT}${NC}"
        return 1
    else
        echo -e "${GREEN}✅ OK: ${var_name} not set to true${NC}"
        return 0
    fi
}

# Track errors
ERRORS=0

# Check based on environment
if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "staging" ]; then
    echo -e "\n${YELLOW}Checking production/staging requirements...${NC}\n"

    # Required variables
    check_required "ENVIRONMENT" || ((ERRORS++))
    check_required "WEBHOOK_SECRET" || ((ERRORS++))
    check_required "UPLOAD_TICKET_SECRET" || ((ERRORS++))

    # Check authentication (at least one method required)
    HAS_API_KEY=${BACKEND_API_KEY:-""}
    HAS_BASIC_USER=${BASIC_AUTH_USERNAME:-""}
    HAS_BASIC_PASS=${BASIC_AUTH_PASSWORD:-""}

    if [ -n "$HAS_API_KEY" ]; then
        echo -e "${GREEN}✅ Auth: API Key configured${NC}"
    elif [ -n "$HAS_BASIC_USER" ] && [ -n "$HAS_BASIC_PASS" ]; then
        echo -e "${GREEN}✅ Auth: Basic Auth configured${NC}"
    else
        echo -e "${RED}❌ Auth: No authentication method configured${NC}"
        echo -e "${RED}   Set either BACKEND_API_KEY or BASIC_AUTH_USERNAME+BASIC_AUTH_PASSWORD${NC}"
        ((ERRORS++))
    fi

    # Forbidden settings
    check_forbidden "AUTH_DISABLED" || ((ERRORS++))
    check_forbidden "DEBUG" || ((ERRORS++))

    # Other required production settings
    echo -e "\n${YELLOW}Checking other production settings...${NC}\n"
    check_required "DATABASE_URL" || ((ERRORS++))
    check_required "R2_ENDPOINT" || ((ERRORS++))
    check_required "R2_ACCESS_KEY" || ((ERRORS++))
    check_required "R2_SECRET_KEY" || ((ERRORS++))
    check_required "R2_BUCKET" || ((ERRORS++))
    check_required "CORS_ORIGINS" || ((ERRORS++))

elif [ "$ENVIRONMENT" = "local" ] || [ "$ENVIRONMENT" = "ci" ]; then
    echo -e "\n${YELLOW}Checking ${ENVIRONMENT} environment...${NC}\n"
    echo -e "${GREEN}✅ Relaxed security allowed for ${ENVIRONMENT}${NC}"

    # Optional but recommended
    if [ "${AUTH_DISABLED:-false}" = "true" ]; then
        echo -e "${YELLOW}⚠️  Warning: AUTH_DISABLED=true (OK for ${ENVIRONMENT})${NC}"
    fi

    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${YELLOW}⚠️  Warning: DEBUG=true (OK for ${ENVIRONMENT})${NC}"
    fi
else
    echo -e "\n${RED}❌ Invalid or missing ENVIRONMENT: ${ENVIRONMENT}${NC}"
    echo -e "${RED}   Must be one of: local, ci, staging, production${NC}"
    ((ERRORS++))
fi

# Summary
echo -e "\n=================================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to deploy.${NC}"
    exit 0
else
    echo -e "${RED}❌ Found ${ERRORS} error(s). Fix them before deploying.${NC}"
    echo -e "\n${YELLOW}Quick fix examples:${NC}"
    echo "export ENVIRONMENT=production"
    echo "export BACKEND_API_KEY=\$(openssl rand -base64 32)"
    echo "export WEBHOOK_SECRET=\$(openssl rand -base64 32)"
    echo "export UPLOAD_TICKET_SECRET=\$(openssl rand -base64 32)"
    exit 1
fi
