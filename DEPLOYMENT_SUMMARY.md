1| # OpsRAG Deployment Summary
2| 
3| ## ✓ Completed
4| 
5| ### 1. Local Development Environment
6| - Docker Compose setup with PostgreSQL + pgvector
7| - All services healthy (backend, frontend, postgres)
8| - API responding correctly with HIGH confidence answers
9| - Synthetic dataset loaded (2,033 sources, 2,619 chunks, 30,587 log lines)
10| 
11| **Test**: Ask the UI at `http://localhost:8501`
12| ```
13| "Why did payment-service fail after deployment v2.8.1?"
14| → Answer: "INC-0406 (SEV1, payment-service): Payment requests returning HTTP 500 after v2.8.1 deploy..."
15| ```
16| 
17| ### 2. GitHub Actions CI/CD Pipeline
18| **File**: `.github/workflows/deploy.yml`
19| 
20| - **Build stage**: Builds backend, frontend, and bootstrap images with Docker BuildKit
21| - **Cache**: Layer caching via registry to speed up subsequent builds
22| - **Push stage**: Tags as `latest` and commit SHA, pushes to Docker Hub
23| - **Test stage**: Runs unit tests + integration tests against PostgreSQL
24| - **Trigger**: Push to `main` branch (auto) or manual workflow dispatch
25| 
26| **Setup required**:
27| ```
28| Go to: https://github.com/yuvi-097/opsrag/settings/secrets/actions
29| Add two secrets:
30|   - DOCKER_HUB_USERNAME: yuvi0011
31|   - DOCKER_HUB_TOKEN: <token from hub.docker.com/settings/security>
32| ```
33| 
34| ### 3. Production Deployment Files
35| - **docker-compose.prod.yml**: Production-ready Compose with:
36|   - Resource limits (4 CPU, 8GB memory)
37|   - Persistent volume for database
38|   - Health checks with longer startup delays
39|   - Proper logging config
40|   - Service startup dependencies (postgres → bootstrap → backend → frontend)
41| 
42| - **.env.prod**: Template for production environment variables
43| 
44| ### 4. Documentation
45| - **DEPLOYMENT.md**: Complete deployment guide with:
46|   - Local dev quick start
47|   - Production deployment steps
48|   - GitHub Actions setup
49|   - Kubernetes examples
50|   - Troubleshooting guide
51|   - Production checklist
52| 
53| ---
54| 
55| ## 🚀 Next Steps
56| 
57| ### Option 1: Push Images to Docker Hub (Recommended First)
58| ```bash
59| # Requires Docker login
60| docker login
61| docker push yuvi0011/opsrag-backend:latest
62| docker push yuvi0011/opsrag-frontend:latest
63| ```
64| 
65| ### Option 2: Trigger GitHub Actions Pipeline
66| ```bash
67| # Push to main branch (or use GitHub UI → Actions → Deploy)
68| git add .github/workflows/deploy.yml DEPLOYMENT.md .env.prod docker-compose.prod.yml
69| git commit -m "Add CI/CD pipeline and production deployment"
70| git push origin main
71| ```
72| → Watch at: `https://github.com/yuvi-097/opsrag/actions`
73| 
74| ### Option 3: Deploy to Production Now
75| ```bash
76| # Edit .env.prod with secure passwords
77| nano .env.prod
78| 
79| # Deploy on Linux/Mac server
80| docker compose -f docker-compose.prod.yml up -d
81| 
82| # Create API token
83| docker compose -f docker-compose.prod.yml exec backend \
84|   python scripts/create_token.py issue alex.rivera --name prod --days 90
85| 
86| # Verify
87| docker compose -f docker-compose.prod.yml ps
88| curl http://localhost:8000/api/ready
89| ```
90| 
91| ---
92| 
93| ## 📋 Quick Reference
94| 
95| ### Local (Dev)
96| ```bash
97| docker compose up                    # Start dev stack
98| docker compose logs -f backend       # Watch logs
99| docker compose down -v               # Clean up
100| ```
101| 
102| ### Production
103| ```bash
104| docker compose -f docker-compose.prod.yml up -d     # Start
105| docker compose -f docker-compose.prod.yml logs -f   # Logs
106| docker compose -f docker-compose.prod.yml down      # Stop
107| ```
108| 
109| ### API Testing
110| ```bash
111| TOKEN="<replace-with-your-token>"
112| curl -X POST http://localhost:8000/api/agent/ask \
113|   -H "Authorization: Bearer $TOKEN" \
114|   -H "Content-Type: application/json" \
115|   -d '{"question": "What caused INC-0406?"}'
116| ```
117| 
118| ### Create API Tokens
119| ```bash
120| # In local dev
121| docker compose exec backend python scripts/create_token.py issue alex.rivera --name dev --days 30
122| 
123| # In production
124| docker compose -f docker-compose.prod.yml exec backend python scripts/create_token.py issue user.name --name reason --days 90
125| ```
126| 
127| ---
128| 
129| ## 🔐 Security Checklist
130| 
131| - [ ] Generated 16+ character secure passwords for POSTGRES_PASSWORD and TOOLS_SQL_PASSWORD
132| - [ ] Set SECURITY_ALLOW_USER_HEADER=false in production
133| - [ ] Credentials are NOT in git (check: `git log --all --oneline -- .env`)
134| - [ ] GitHub Actions secrets configured (DOCKER_HUB_USERNAME, DOCKER_HUB_TOKEN)
135| - [ ] API tokens created and stored securely
136| - [ ] Non-root user in containers (uid 10001, 10002)
137| - [ ] Health checks configured with appropriate timeouts
138| - [ ] Resource limits set (memory 8GB, CPU 4)
139| 
140| ---
141| 
142| ## 📊 Performance Notes
143| 
144| ### Current Metrics (CPU-only laptop)
145| - Answer latency: p50 656ms, p95 2.2s, p99 2.7s
146| - Bootstrap time: ~313s (embedding 2,619 chunks on CPU)
147| - Throughput: ~0.95 requests/sec, ~1.3 with 8 concurrent
148| - Models baked into image (~4.5GB backend image)
149| 
150| ### Scaling
151| - Single instance handles ~1-1.3 answers/sec on CPU
152| - For higher throughput: add GPU or multi-worker setup with load balancer
153| - Database is not a bottleneck (tool queries ~ms)
154| - Primary cost: model inference (52% reranker, 32% verification)
155| 
156| ---
157| 
158| ## 📚 Files Created/Modified
159| 
160| ```
161| opsrag/
162| ├── .github/workflows/deploy.yml         # GitHub Actions CI/CD
163| ├── docker-compose.prod.yml              # Production Compose config
164| ├── .env.prod                            # Production env template
165| ├── DEPLOYMENT.md                        # Deployment guide
166| └── README.md                            # (existing, unchanged)
167| ```
168| 
169| ---
170| 
171| ## 🎯 What's Deployed
172| 
173| | Component | Status | Version | Location |
174| |-----------|--------|---------|----------|
175| | Backend API | ✓ Running | e31f78c8d10a | localhost:8000 |
176| | Frontend UI | ✓ Running | c394aabd5c3c | localhost:8501 |
177| | PostgreSQL + pgvector | ✓ Running | pg17 | localhost:5433 |
178| | Synthetic dataset | ✓ Loaded | seed 42 | pgdata volume |
179| 
180| **Total deployment size**: ~5.3GB (backend 4.5GB + frontend 800MB)
181| 
182| ---
183| 
184| ## 💬 Questions?
185| 
186| - Deployment guide: `cat DEPLOYMENT.md`
187| - CI/CD pipeline: `.github/workflows/deploy.yml`
188| - Logs: `docker compose logs -f [service]`
189| - API docs: `http://localhost:8000/api/docs`
190| 