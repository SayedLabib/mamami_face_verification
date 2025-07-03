#!/bin/bash
# dns_test.sh - Script for testing DNS resolution and API connectivity inside the container

echo "=== DNS Resolution & API Connectivity Test ==="
echo ""

echo "Testing DNS resolution for key hosts..."
for host in api-us.faceplusplus.com google.com qdrant db; do
    echo -n "Resolving $host: "
    if getent hosts $host > /dev/null; then
        ip=$(getent hosts $host | awk '{ print $1 }')
        echo "SUCCESS ($ip)"
    else
        echo "FAILED"
    fi
done

echo ""
echo "Testing API connectivity..."
echo -n "HTTPS request to api-us.faceplusplus.com: "
if curl -s --connect-timeout 5 -I https://api-us.faceplusplus.com > /dev/null; then
    echo "SUCCESS"
else
    echo "FAILED"
    echo -n "Testing direct IP access: "
    if curl -s --connect-timeout 5 -I http://104.21.23.75/health > /dev/null; then
        echo "SUCCESS"
    else
        echo "FAILED"
    fi
fi

echo ""
echo "Testing Qdrant connectivity..."
echo -n "HTTP request to Qdrant: "
if curl -s --connect-timeout 5 -I http://qdrant:6333/collections > /dev/null; then
    echo "SUCCESS"
else
    echo "FAILED"
fi

echo ""
echo "Checking DNS configuration..."
if [ -f /app/dns_config/resolv.conf ]; then
    echo "Custom DNS configuration:"
    cat /app/dns_config/resolv.conf
else
    echo "No custom DNS config found"
fi
echo ""
echo "System DNS configuration:"
cat /etc/resolv.conf

echo ""
echo "Extra hosts configuration:"
cat /etc/hosts | grep -v "^#" | grep -v "^$"

echo ""
echo "=== Test Complete ==="
