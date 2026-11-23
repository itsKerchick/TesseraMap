# Развёртывание TesseraMap: Архитектура Клиент (VPS) / Домашний Сервер (Worker)

Данная архитектура решает задачу:
1. **Сайт доступен 24/7**, даже когда домашний компьютер выключен или спит. Пользователи всегда могут зайти на `https://tessera.mrkerchick.ru`, открыть спутниковую карту, ввести свой MapTiler API ключ, читать туториал.
2. **Домашний ПК не требует белого IP, статического IP или проброса портов (NAT/CGNAT traversal)**: домашний воркер сам открывает исходящее долгоживущее WebSocket-соединение (`/ws/worker`) к VPS в Германии. Через этот защищённый туннель VPS проксирует запросы на сегментацию и получает телеметрию GPU.

---

## Схема взаимодействия

```
  [ Пользователь в браузере ]
              │  HTTPS GET / (Сайт открывается ВСЕГДА)
              ▼
    ┌──────────────────────────────────────────────┐
    │          VPS в Германии (Белый IP)           │
    │  Веб-сервер: Caddy (авто-SSL Let's Encrypt)  │
    │  Шлюз: /home/sites/TesseraMap/vps_server.py  │
    │  • Раздача index.html, guide, favicon, DOCX  │
    │  • Проверка ключей MapTiler и геокодинг      │
    │  • WebSocket Gateway: /ws/worker             │
    └──────────────────────┬───────────────────────┘
                           ▲
                           │ Исходящий WebSocket-туннель
                           │ (через любой серый IP / NAT)
                           │ wss://tessera.mrkerchick.ru/ws/worker
                           ▼
    ┌──────────────────────────────────────────────┐
    │        Домашний ПК (Серый IP / NAT)          │
    │  Файл: worker.py                             │
    │  • GPU: NVIDIA GeForce RTX 4060 Ti (CUDA)    │
    │  • PyTorch: SegFormer + YOLO + Ансамбль     │
    │  • Мониторинг NVML (нагрузка GPU, VRAM)      │
    │  • Автоматический реконнект при обрывах      │
    └──────────────────────────────────────────────┘
```

---

## 1. Что закинуть на VPS (Debian 13)

На VPS **НЕ** нужны видеокарта, CUDA или тяжёлый PyTorch. Достаточно обычного недорогого сервера (1 vCPU, 512 MB – 1 GB RAM).

Файлы размещаются в каталоге: `/home/sites/TesseraMap`

Скопируйте следующие файлы из папки `TesseraMap`:
- `vps_server.py` — веб-сервер и WebSocket-шлюз
- `index.html` — главное SPA-приложение карты
- `guide.html` — веб-гайд
- `maptiler_guide.html` — онлайн-просмотрщик DOCX
- `maptiler_guide.docx` — документ Word с инструкцией
- `maptiler_guide.pdf` — PDF версия инструкции
- `guide_assets/` — папка с иллюстрациями гайда
- `favicon.ico`, `favicon.png`, `favicon.svg` — фавиконки

> **Быстрая отправка с локального ПК через PowerShell (SCP):**
> ```powershell
> scp -r "L:\Local project 2\TesseraMap\vps_server.py" "L:\Local project 2\TesseraMap\index.html" "L:\Local project 2\TesseraMap\guide.html" "L:\Local project 2\TesseraMap\maptiler_guide.html" "L:\Local project 2\TesseraMap\maptiler_guide.docx" "L:\Local project 2\TesseraMap\maptiler_guide.pdf" "L:\Local project 2\TesseraMap\guide_assets" "L:\Local project 2\TesseraMap\favicon.*" root@IP_ВАШЕГО_СЕРВЕРА:/home/sites/TesseraMap/
> ```

---

## 2. Установка окружения на сервере (Debian 13)

В Debian 12/13 действует ограничение PEP 668 на установку пакетов напрямую в систему, поэтому используем изолированный `venv`:

```bash
mkdir -p /home/sites/TesseraMap
cd /home/sites/TesseraMap

# Установка Python и venv
apt update && apt install -y python3 python3-venv python3-pip

# Создание виртуального окружения и установка зависимостей
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install fastapi "uvicorn[standard]" httpx websockets pydantic
```

---

## 3. Настройка системного сервиса Systemd

Создайте файл службы `/etc/systemd/system/tesseramap.service`:

```bash
cat << 'EOF' > /etc/systemd/system/tesseramap.service
[Unit]
Description=TesseraMap Web Server & ML Gateway
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/sites/TesseraMap
Environment=PORT=7861
Environment=HOST=127.0.0.1
Environment=WORKER_SECRET=tessera-secret-2026
ExecStart=/home/sites/TesseraMap/.venv/bin/python vps_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
```

---

## 4. Настройка Caddy (Reverse Proxy + Авто-SSL)

В **Caddy** поддержка WebSocket встроена по умолчанию в директиву `reverse_proxy`. Caddy также автоматически получает и продлевает бесплатные сертификаты Let's Encrypt / ZeroSSL.

Откройте `/etc/caddy/Caddyfile`:
```bash
nano /etc/caddy/Caddyfile
```

Добавьте блок для вашего домена `tessera.mrkerchick.ru`:

```caddyfile
tessera.mrkerchick.ru {
    reverse_proxy 127.0.0.1:7861
}
```

Примените изменения в Caddy:
```bash
systemctl reload caddy
# или: caddy reload --config /etc/caddy/Caddyfile
```

---

## 5. Создание команд `tessera start` / `tessera stop`

Чтобы управлять сервисом через удобные команды `tessera start`, `tessera stop` и т.д., создадим исполняемый скрипт в `/usr/local/bin/tessera`:

```bash
cat << 'EOF' > /usr/local/bin/tessera
#!/usr/bin/env bash

case "$1" in
  start)
    echo "Запуск TesseraMap..."
    systemctl start tesseramap
    systemctl is-active --quiet tesseramap && echo "✓ TesseraMap успешно запущен!" || echo "✗ Ошибка запуска"
    ;;
  stop)
    echo "Остановка TesseraMap..."
    systemctl stop tesseramap
    echo "✓ TesseraMap остановлен."
    ;;
  restart)
    echo "Перезапуск TesseraMap..."
    systemctl restart tesseramap
    systemctl is-active --quiet tesseramap && echo "✓ TesseraMap перезапущен!" || echo "✗ Ошибка"
    ;;
  status)
    systemctl status tesseramap --no-pager
    ;;
  logs)
    journalctl -u tesseramap -f -n 50
    ;;
  *)
    echo "Использование: tessera {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
EOF

chmod +x /usr/local/bin/tessera
```

### Доступные команды:
- `tessera start` — запустить сервер
- `tessera stop` — остановить сервер
- `tessera restart` — перезапустить
- `tessera status` — проверить статус и время работы
- `tessera logs` — смотреть живой поток логов в реальном времени

---

## 6. Запуск домашнего ML-воркера (на вашем ПК)

На домашнем компьютере у вас уже установлены PyTorch, CUDA, SegFormer и YOLO в `.venv`.

1. В файле `run_worker.bat` укажите адрес вашего сервера через защищённый протокол `wss://`:
   ```cmd
   set VPS_URL=wss://tessera.mrkerchick.ru/ws/worker
   set WORKER_SECRET=tessera-secret-2026
   ```

2. Запустите `run_worker.bat`:
   ```
   === УСПЕШНО ПОДКЛЮЧЕНО К VPS ===
   Домашний сервер готов принимать запросы на сегментацию от пользователей!
   ```

---

## 7. Полная деинсталляция (Удаление с сервера)

Если потребуется полностью и чисто удалить TesseraMap с вашего сервера:

```bash
# 1. Остановить сервис и отключить его автозапуск
tessera stop
systemctl disable tesseramap

# 2. Удалить файл системной службы systemd
rm -f /etc/systemd/system/tesseramap.service
systemctl daemon-reload

# 3. Удалить исполняемую команду tessera
rm -f /usr/local/bin/tessera

# 4. Удалить блок tessera.mrkerchick.ru из /etc/caddy/Caddyfile
# Откройте Caddyfile и удалите блок:
# tessera.mrkerchick.ru {
#     reverse_proxy 127.0.0.1:7861
# }
# Затем перезагрузите Caddy:
systemctl reload caddy

# 5. Удалить каталог проекта и виртуальное окружение
rm -rf /home/sites/TesseraMap
```

После этих шагов на сервере не останется никаких фоновых процессов, служб или остаточных файлов.
