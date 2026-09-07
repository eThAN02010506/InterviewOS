const appleReleaseVariables = [
  'APPLE_SIGNING_IDENTITY',
  'APPLE_CERTIFICATE',
  'APPLE_CERTIFICATE_PASSWORD',
  'APPLE_API_ISSUER',
  'APPLE_API_KEY',
  'APPLE_API_KEY_PATH',
  'APPLE_ID',
  'APPLE_PASSWORD',
  'APPLE_TEAM_ID'
];

if (process.platform === 'darwin') {
  const configured = appleReleaseVariables.filter(name => String(process.env[name] || '').trim());
  if (configured.length) {
    console.error(
      `Local build refused: unset Apple release credentials (${configured.join(', ')}) before ` +
      'building the ad-hoc bundle. This prevents a distributable identity from signing the ' +
      'local-only library-validation entitlement.'
    );
    process.exit(1);
  }
}
