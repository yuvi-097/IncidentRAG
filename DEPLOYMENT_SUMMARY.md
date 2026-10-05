# OpsRAG Deployment Summary

## ✓ Completed

### 1. Local Development Environment
- Docker Compose setup with PostgreSQL + pgvector
- All services healthy (backend, frontend, postgres)
- API responding correctly with HIGH confidence answers
- Synthetic dataset loaded (2,033 sources, 2,619 chunks, 30,587 log lines)

**Test**: Ask the UI at `http://localhost:8501`
```
"Why did payment-service fail after deployment v2.8.1?"
→ Answer: "INC-0406 (SEV1, payment-service): Payment requests returning HTTP 500 after v2.8.1 deploy..."
```

### 2. GitHub Actions CI/CD Pipeline
**File**: `.github/workflows/deploy.yml`

- **Build stage**: Builds backend, frontend, and bootstrap images with Docker BuildKit
- **Cache**: Layer caching via registry to speed up subsequent builds
- **Push stage**: Tags as `latest` and commit SHA, pushes to Docker Hub
- **Test stage**: Runs unit tests + integration tests against PostgreSQL
- **Trigger**: Push to `main` branch (auto) or manual workflow dispatch

**Setup required**:
```
Go to: https://github.com/yuvi-097/opsrag/settings/secrets/actions
Add two secrets:
  - DOCKER_HUB_USERNAME: yuvi0011
  - DOCKER_HUB_TOKEN: <token from hub.docker.com/settings/security>
```

### 3. Production Deployment Files
- **docker-compose.prod.yml**: Production-ready Compose with:
  - Resource limits (4 CPU, 8GB memory)
  - Persistent volume for database
  - Health checks with longer startup delays
  - Proper logging config
  - Service startup dependencies (postgres → bootstrap → backend → frontend)

- **.env.prod**: Template for production environment variables

### 4. Documentation
- **DEPLOYMENT.md**: Complete deployment guide with:
  - Local dev quick start
  - Production deployment steps
  - GitHub Actions setup
  - Kubernetes examples
  - Troubleshooting guide
  - Production checklist

---

## 🚀 Next Steps

### Option 1: Push Images to Docker Hub (Recommended First)
```bash
# Requires Docker login
docker login
docker push yuvi0011/opsrag-backend:latest
docker push yuvi0011/opsrag-frontend:latest
```

### Option 2: Trigger GitHub Actions Pipeline
```bash
# Push to main branch (or use GitHub UI → Actions → Deploy)
git add .github/workflows/deploy.yml DEPLOYMENT.md .env.prod docker-compose.prod.yml
git commit -m "Add CI/CD pipeline and production deployment"
git push origin main
```
→ Watch at: `https://github.com/yuvi-097/opsrag/actions`

### Option 3: Deploy to Production Now
```bash
# Edit .env.prod with secure passwords
nano .env.prod

# Deploy on Linux/Mac server
docker compose -f docker-compose.prod.yml up -d

# Create API token
docker compose -f docker-compose.prod.yml exec backend \
  python scripts/create_token.py issue alex.rivera --name prod --days 90

# Verify
docker compose -f docker-compose.prod.yml ps
curl http://localhost:8000/api/ready
```

---

## 📋 Quick Reference

### Local (Dev)
```bash
docker compose up                    # Start dev stack
docker compose logs -f backend       # Watch logs
docker compose down -v               # Clean up
```

### Production
```bash
docker compose -f docker-compose.prod.yml up -d     # Start
docker compose -f docker-compose.prod.yml logs -f   # Logs
docker compose -f docker-compose.prod.yml down      # Stop
```

### API Testing
```bash
TOKEN="REPLACE_WITH_TOKEN"
curl -X POST http://localhost:8000/api/agent/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What caused INC-0406?"}'
```

### Create API Tokens
```bash
# In local dev
docker compose exec backend python scripts/create_token.py issue alex.rivera --name dev --days 30

# In production
docker compose -f docker-compose.prod.yml exec backend python scripts/create_token.py issue user.name --name reason --days 90
```

---

## 🔐 Security Checklist

- [ ] Generated 16+ character secure passwords for POSTGRES_PASSWORD and TOOLS_SQL_PASSWORD
- [ ] Set SECURITY_ALLOW_USER_HEADER=false in production
- [ ] Credentials are NOT in git (check: `git log --all --oneline -- .env`)
- [ ] GitHub Actions secrets configured (DOCKER_HUB_USERNAME, DOCKER_HUB_TOKEN)
- [ ] API tokens created and stored securely
- [ ] Non-root user in containers (uid 10001, 10002)
- [ ] Health checks configured with appropriate timeouts
- [ ] Resource limits set (memory 8GB, CPU 4)

---

## 📊 Performance Notes

### Current Metrics (CPU-only laptop)
- Answer latency: p50 656ms, p95 2.2s, p99 2.7s
- Bootstrap time: ~313s (embedding 2,619 chunks on CPU)
- Throughput: ~0.95 requests/sec, ~1.3 with 8 concurrent
- Models baked into image (~4.5GB backend image)

### Scaling
- Single instance handles ~1-1.3 answers/sec on CPU
- For higher throughput: add GPU or multi-worker setup with load balancer
- Database is not a bottleneck (tool queries ~ms)
- Primary cost: model inference (52% reranker, 32% verification)

---

## 📚 Files Created/Modified

```
opsrag/
├── .github/workflows/deploy.yml         # GitHub Actions CI/CD
├── docker-compose.prod.yml              # Production Compose config
├── .env.prod                            # Production env template
├── DEPLOYMENT.md                        # Deployment guide
└── README.md                            # (existing, unchanged)
```

---

## 🎯 What's Deployed

| Component | Status | Version | Location |
|-----------|--------|---------|----------|
| Backend API | ✓ Running | e31f78c8d10a | localhost:8000 |
| Frontend UI | ✓ Running | c394aabd5c3c | localhost:8501 |
| PostgreSQL + pgvector | ✓ Running | pg17 | localhost:5433 |
| Synthetic dataset | ✓ Loaded | seed 42 | pgdata volume |

**Total deployment size**: ~5.3GB (backend 4.5GB + frontend 800MB)

---

## 💬 Questions?

- Deployment guide: `cat DEPLOYMENT.md`
- CI/CD pipeline: `.github/workflows/deploy.yml`
- Logs: `docker compose logs -f [service]`
- API docs: `http://localhost:8000/api/docs`
