FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .

EXPOSE 5173

# В compose VITE_API_BASE проставляется через env
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
