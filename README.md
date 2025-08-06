


Following `Newline - Responsive LLM Applications with Server-Sent Events`

```bash
pipenv install "numpy<2"
```

Installing mitmproxy

```
python3 -m pip install pipx
which pipx
pipx install mitmproxy
mitmweb
export HTTP_PROXY=http://127.0.0.1:8080
export HTTPS_PROXY=http://127.0.0.1:8080
```


Start mongo db in docker
```
docker volume create mongodb_data
docker run -d -p 27017:27017 -v mongodb_data:/data/db \
-e MONGO_INITDB_ROOT_USERNAME=admin \
-e MONGO_INITDB_ROOT_PASSWORD=your_password \
--name mongodb-container mongo:latest
```