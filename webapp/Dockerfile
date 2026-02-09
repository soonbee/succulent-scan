FROM dunglas/frankenphp AS builder

RUN install-php-extensions pdo_sqlite zip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Node.js 22 설치
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /app

# PHP 의존성 설치
COPY composer.json composer.lock ./
RUN composer install --no-dev --no-interaction --no-scripts --prefer-dist --optimize-autoloader

# Node 의존성 설치
COPY package.json package-lock.json ./
RUN npm ci

# 소스 복사 및 빌드
COPY . .
RUN composer run-script post-autoload-dump
RUN npm run build

# 프로덕션에 불필요한 파일 제거
RUN rm -rf node_modules tests

FROM dunglas/frankenphp

RUN install-php-extensions \
    pdo_sqlite \
    gd \
    intl \
    zip \
    opcache \
    pcntl \
    bcmath

WORKDIR /app

# 빌더에서 앱 전체 복사
COPY --from=builder /app .

COPY Caddyfile /etc/caddy/Caddyfile

# 쓰기 가능한 디렉토리 설정
RUN mkdir -p storage/app/public \
    storage/framework/cache \
    storage/framework/sessions \
    storage/framework/views \
    storage/logs \
    database \
    bootstrap/cache \
    && chmod -R 775 storage bootstrap/cache database \
    && touch database/database.sqlite \
    && php artisan storage:link

ENV APP_ENV=production \
    APP_DEBUG=false \
    DB_CONNECTION=sqlite \
    LOG_CHANNEL=stderr

EXPOSE 80

ENTRYPOINT ["sh", "-c", "php artisan migrate --force && frankenphp run --config /etc/caddy/Caddyfile"]
