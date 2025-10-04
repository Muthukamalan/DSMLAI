# MCP

## Install Nodejs
```bash
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.19.0".

# Verify npm version:
npm -v # Should print "10.9.3".

```

# Three MCP Transports
1. STUDIO 
    - Client launches server as child process
    - Messages sent  via server's stdin
    - Responses received from server's stdout
    - No network overhead - direct memory access
    - Inherently secure
    - same machine (local server only)
    - single server instance per client 
2. Server Sent Events
    - HTTP Post for client-to-client requests
    - SSE stream for server-to-client responses
    - Automatic reconnection on connection loss
    - works across networks and browsers
    - more complex than studio 
    - browser connection limits 
3. Streamable HTTP 
    - HTTP/HTTPS with chuncked transfer encoding
    - Built in SSL/TLS security


# SHell
```sh
DANGEROUSLY_OMIT_AUTH=true uv run mcp dev main.py 
```