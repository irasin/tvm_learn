# set_proxy

科学上网有很多种方法，VPN一般可以解决很多问题

在WSL2上使用wget时遇到了下述的问题

```bash
$ wget https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/cuda_v0.10.log

--2022-10-10 22:41:42--  https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/cuda_v0.10.log
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 0.0.0.0, ::
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|0.0.0.0|:443... failed: Connection refused.
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|::|:443... failed: Connection refused.
```

同时python中也有
```bash
>>> import urllib.request as urllib2
>>> urllib2.urlretrieve("https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/cuda_v0.10.log", "log.txt")
Traceback (most recent call last):
  File "/usr/lib/python3.8/urllib/request.py", line 1354, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/usr/lib/python3.8/http/client.py", line 1256, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/usr/lib/python3.8/http/client.py", line 1302, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/usr/lib/python3.8/http/client.py", line 1251, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/usr/lib/python3.8/http/client.py", line 1011, in _send_output
    self.send(msg)
  File "/usr/lib/python3.8/http/client.py", line 951, in send
    self.connect()
  File "/usr/lib/python3.8/http/client.py", line 1418, in connect
    super().connect()
  File "/usr/lib/python3.8/http/client.py", line 922, in connect
    self.sock = self._create_connection(
  File "/usr/lib/python3.8/socket.py", line 808, in create_connection
    raise err
  File "/usr/lib/python3.8/socket.py", line 796, in create_connection
    sock.connect(sa)
ConnectionRefusedError: [Errno 111] Connection refused

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.8/urllib/request.py", line 247, in urlretrieve
    with contextlib.closing(urlopen(url, data)) as fp:
  File "/usr/lib/python3.8/urllib/request.py", line 222, in urlopen
    return opener.open(url, data, timeout)
  File "/usr/lib/python3.8/urllib/request.py", line 525, in open
    response = self._open(req, data)
  File "/usr/lib/python3.8/urllib/request.py", line 542, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "/usr/lib/python3.8/urllib/request.py", line 502, in _call_chain
    result = func(*args)
  File "/usr/lib/python3.8/urllib/request.py", line 1397, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "/usr/lib/python3.8/urllib/request.py", line 1357, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 111] Connection refused>
>>> quit()

```

原因是因为github的ip问题，所以可以采取[这里](https://blog.csdn.net/qq_28382661/article/details/111192651)的方法，

1. 打开 https://www.ipaddress.com/ 输入访问不了的域名 raw.githubusercontent.com，找到ip地址，比如如下

```text

What is raw.githubusercontent.com IP address?
raw.githubusercontent.com resolves to 4 IPv4 addresses and 4 IPv6 addresses:
185.199.108.133
185.199.109.133
185.199.110.133
185.199.111.133
2606:50c0:8000::154
2606:50c0:8001::154
2606:50c0:8002::154
2606:50c0:8003::154
```

2. 添加对应的ip，可以直接执行下述，或者直接打开`/etc/hosts`手动添加

```bash
sudo bash -c 'echo "185.199.108.133 raw.githubusercontent.com" >> /etc/hosts'
sudo bash -c 'echo "185.199.109.133 raw.githubusercontent.com" >> /etc/hosts'
sudo bash -c 'echo "185.199.110.133 raw.githubusercontent.com" >> /etc/hosts'
sudo bash -c 'echo "185.199.111.133 raw.githubusercontent.com" >> /etc/hosts'

sudo bash -c 'echo "2606:50c0:8000::154 raw.githubusercontent.com" >> /etc/hosts'
sudo bash -c 'echo "2606:50c0:8001::154 raw.githubusercontent.com" >> /etc/hosts'
sudo bash -c 'echo "2606:50c0:8002::154 raw.githubusercontent.com" >> /etc/hosts'
sudo bash -c 'echo "2606:50c0:8003::154 raw.githubusercontent.com" >> /etc/hosts'

```
