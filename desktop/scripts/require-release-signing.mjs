import {spawnSync} from 'node:child_process';

if (process.platform === 'darwin') {
  const identity = String(process.env.APPLE_SIGNING_IDENTITY || '').trim();
  if (!identity.startsWith('Developer ID Application:')) {
    console.error(
      'Release build refused: APPLE_SIGNING_IDENTITY must name a Developer ID Application identity. ' +
      'Use `npm run build:local` for an ad-hoc local test bundle.'
    );
    process.exit(1);
  }
  const identities = spawnSync('security', ['find-identity', '-v', '-p', 'codesigning'], {encoding: 'utf8'});
  if (identities.status !== 0 || !String(identities.stdout || '').includes(identity)) {
    console.error('Release build refused: APPLE_SIGNING_IDENTITY was not found in the login keychain.');
    process.exit(1);
  }
  const apiNotarization = String(process.env.APPLE_API_ISSUER || '').trim() &&
    (String(process.env.APPLE_API_KEY || '').trim() || String(process.env.APPLE_API_KEY_PATH || '').trim());
  const appleIdNotarization = String(process.env.APPLE_ID || '').trim() &&
    String(process.env.APPLE_PASSWORD || '').trim() && String(process.env.APPLE_TEAM_ID || '').trim();
  if (!apiNotarization && !appleIdNotarization) {
    console.error('Release build refused: configure App Store Connect API credentials or APPLE_ID/APPLE_PASSWORD/APPLE_TEAM_ID for notarization.');
    process.exit(1);
  }
}
