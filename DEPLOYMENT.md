# OpsRAG Deployment Guide

## Quick Start (Docker Compose)

### Local Development
```bash
cp .env.example .env
# Edit .env with secure passwords (16+ characters)
docker compose up --build --pull always
```

Access: `http://localhost:8501` (UI) and `http://localhost:8000` (API)

### Production Deployment

#### 1. Generate Secure Passwords
```bash
python -c "import secrets; print(secrets.token_urlsafe(24)); print(secrets.token_urlsafe(24))"
```

#### 2. Create `.env.prod`
```bash
cp .env.example .env.prod
# Edit with secure values:
# POSTGRES_PASSWORD=<secure-password>
# TOOLS_SQL_PASSWORD=<secure-password>
# DOCKER_HUB_USERNAME=yuvi0011
# IMAGE_TAG=latest
```

#### 3. Pull Latest Images and Start
```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

#### 4. Create API Token
```bash
docker compose -f docker-compose.prod.yml exec backend \
  python scripts/create_token.py issue alex.rivera --name prod --days 90
```

#### 5. Verify Services
```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend
```

---

## GitHub Actions CI/CD Setup

### Prerequisites
- Push `.github/workflows/deploy.yml` to GitHub
- Set secrets in GitHub repo settings (`Settings > Secrets and variables > Actions`)

### Required Secrets
1. **DOCKER_HUB_USERNAME**: `yuvi0011`
2. **DOCKER_HUB_TOKEN**: [Personal Access Token from Docker Hub](https://hub.docker.com/settings/security)

### How it Works
- **Trigger**: Push to `main` branch or workflow dispatch
- **Build**: Builds backend, frontend, and bootstrap images with BuildKit caching
- **Push**: Tags as `latest` and commit SHA, pushes to Docker Hub
- **Test**: Runs unit and integration tests against PostgreSQL
- **Notify**: Reports success/failure

### View Results
- GitHub Actions: `https://github.com/yuvi-097/opsrag/actions`
- Docker Hub: `https://hub.docker.com/r/yuvi0011/opsrag-backend`

---

## Kubernetes Deployment

### Prerequisites
- Helm or kubectl
- PostgreSQL with pgvector (use [Bitnami PostgreSQL chart](https://artifacthub.io/packages/helm/bitnami/postgresql))

### Quick Kubernetes Deploy
```bash
kubectl create namespace opsrag
kubectl -n opsrag create secret generic opsrag-secrets \
  --from-literal=postgres-password=<secure-password> \
  --from-literal=sql-reader-password=<secure-password>

kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: opsrag-backend
  namespace: opsrag
spec:
  selector:
    app: opsrag-backend
  ports:
    - port: 8000
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opsrag-backend
  namespace: opsrag
spec:
  replicas: 1
  selector:
    matchLabels:
      app: opsrag-backend
  template:
    metadata:
      labels:
        app: opsrag-backend
    spec:
      containers:
      - name: backend
        image: yuvi0011/opsrag-backend:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: POSTGRES_HOST
          value: postgres-postgresql
        - name: POSTGRES_PORT
          value: "5432"
        - name: POSTGRES_USER
          value: opsrag
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: opsrag-secrets
              key: postgres-password
        - name: OPSRAG_ENVIRONMENT
          value: production
        - name: OPSRAG_PRELOAD_AGENT
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/ready
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 10
EOF
```

---

## Monitoring and Logs

### Docker Compose
```bash
# View logs
docker compose logs -f backend

# Monitor resource usage
docker compose stats

# Inspect container
docker compose exec backend python -c "import app; print(app.__version__)"
```

### Health Checks
```bash
# API ready check
curl http://localhost:8000/api/ready

# API health
curl http://localhost:8000/api/health

# UI health
curl http://localhost:8501/_stcore/health
```

---

## Production Checklist

- [ ] Set strong passwords (16+ characters)
- [ ] Enable TLS/HTTPS with reverse proxy (nginx/Caddy)
- [ ] Set up database backups
- [ ] Configure log aggregation (ELK, Datadog, CloudWatch)
- [ ] Set resource limits (memory, CPU)
- [ ] Enable health checks and auto-restart
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Create API tokens with expiration
- [ ] Test backup and restore procedures
- [ ] Document runbook for incidents

---

## Troubleshooting

**Containers won't start:**
```bash
docker compose logs bootstrap
docker compose logs backend
```

**Database connection errors:**
```bash
docker compose exec postgres psql -U opsrag -d opsrag -c "SELECT version();"
```

**API returns 503 (models not ready):**
- Check backend logs: `docker compose logs -f backend`
- Wait for model preload to complete (up to 3 minutes)

**Port already in use:**
Edit `.env.prod`:
```
OPSRAG_API_PORT=8001
OPSRAG_UI_PORT=8502
POSTGRES_PUBLISH_PORT=5433
```

---

## Cleanup

**Stop without deleting data:**
```bash
docker compose -f docker-compose.prod.yml down
```

**Stop and delete everything:**
```bash
docker compose -f docker-compose.prod.yml down -v
```

---

## Next Steps

1. **Verify deployment**: Ask the UI a question to test end-to-end
2. **Set up monitoring**: Add Prometheus scrape targets to backend metrics endpoint
3. **Configure backups**: Regular PostgreSQL snapshots
4. **Scale horizontally**: Add more backend replicas with a load balancer
5. **Set up CI/CD alerts**: Slack/email notifications on GitHub Actions failures
