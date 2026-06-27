/**
 * Generate a self-signed TLS certificate for local HTTPS development.
 *
 * The certificate includes Subject Alternative Names (SANs) for:
 *   - localhost
 *   - 127.0.0.1
 *   - the machine's current LAN IP (auto-detected)
 *
 * Output:  .cert/key.pem  +  .cert/cert.pem
 *
 * Usage:   node scripts/generate-cert.mjs
 */

import selfsigned from 'selfsigned';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const certDir = path.join(projectRoot, '.cert');

function getLanIPs() {
  const ips = [];
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name] || []) {
      if (iface.family === 'IPv4' && !iface.internal) {
        ips.push(iface.address);
      }
    }
  }
  return ips;
}

const lanIPs = getLanIPs();
console.log('Detected LAN IP(s):', lanIPs.length ? lanIPs.join(', ') : 'none');

const altNames = [
  { type: 2, value: 'localhost' },
  { type: 7, value: '127.0.0.1' },
];
for (const ip of lanIPs) {
  altNames.push({ type: 7, value: ip });
}

const pems = await selfsigned.generate(
  [{ name: 'commonName', value: 'DepthViz Dev Server' }],
  {
    keySize: 2048,
    days: 365,
    algorithm: 'sha256',
    extensions: [{ name: 'subjectAltName', altNames }],
  }
);

fs.mkdirSync(certDir, { recursive: true });
fs.writeFileSync(path.join(certDir, 'key.pem'), pems.private + '\n');
fs.writeFileSync(path.join(certDir, 'cert.pem'), pems.cert + '\n');

console.log(`\nCertificates written to ${path.relative(projectRoot, certDir)}/`);
console.log('  key.pem');
console.log('  cert.pem');

if (lanIPs.length) {
  console.log('\n--- Next steps ---');
  console.log('1. Run:  npm run dev:https');
  console.log(`2. Open on this machine:    https://localhost:3000`);
  console.log(`3. Open on other devices:   https://${lanIPs[0]}:3000`);
  console.log('\n  ⚠  Browser will warn "Your connection is not private".');
  console.log('     Click Advanced → Proceed anyway. This is normal for self-signed certs.');
}
