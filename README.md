## Single task

```
read from stream buffered + mimalloc + http2

1943ms per 10 reqests
2096ms per 10 reqests
1925ms per 10 reqests
1932ms per 10 reqests
1948ms per 10 reqests

read from stream buffered + http2

1985ms per 10 reqests
1980ms per 10 reqests
1901ms per 10 reqests
1917ms per 10 reqests

read from stream buffered + mimalloc + http1

1349ms per 10 reqests
1340ms per 10 reqests
1356ms per 10 reqests
1306ms per 10 reqests

read from stream unbuffered + mimalloc + http1

1336ms per 10 reqests
1342ms per 10 reqests
1337ms per 10 reqests
1318ms per 10 reqests

read from buffer owned + mimalloc + http1

1637ms per 10 reqests
1646ms per 10 reqests
1645ms per 10 reqests
1626ms per 10 reqests


read from stream unbuffered + mimalloc + http1 + 2MB huge page

748ms per 10 reqests
731ms per 10 reqests
755ms per 10 reqests
760ms per 10 reqests
736ms per 10 reqests

read from stream unbuffered + mimalloc + http1 + 1GB huge page

701ms per 10 reqests
698ms per 10 reqests
698ms per 10 reqests
700ms per 10 reqests
702ms per 10 reqests
702ms per 10 reqests
```
