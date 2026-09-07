import {existsSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const desktopDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repositoryDir = resolve(desktopDir, '..');
const requestedScript = process.argv[2];

if (!requestedScript) {
  console.error('usage: node run-python.mjs <script> [...args]');
  process.exit(2);
}

const virtualEnvironmentPython = process.platform === 'win32'
  ? resolve(repositoryDir, '.venv', 'Scripts', 'python.exe')
  : resolve(repositoryDir, '.venv', 'bin', 'python');
const configuredPython = process.env.INTERVIEW_OS_BUILD_PYTHON?.trim();
if (configuredPython && !existsSync(configuredPython)) {
  console.error(`INTERVIEW_OS_BUILD_PYTHON does not exist: ${configuredPython}`);
  process.exit(2);
}
const candidates = [
  ...(configuredPython ? [[configuredPython]] : []),
  ...(existsSync(virtualEnvironmentPython) ? [[virtualEnvironmentPython]] : []),
  ...(process.platform === 'win32' ? [['py', '-3'], ['python']] : [['python3'], ['python']])
];

for (const [command, ...prefixArgs] of candidates) {
  const result = spawnSync(
    command,
    [...prefixArgs, resolve(desktopDir, requestedScript), ...process.argv.slice(3)],
    {cwd: repositoryDir, stdio: 'inherit'}
  );
  if (!result.error) process.exit(result.status ?? 1);
  if (result.error.code !== 'ENOENT') {
    console.error(`failed to launch Python: ${result.error.message}`);
    process.exit(1);
  }
}

console.error('Python was not found. Set INTERVIEW_OS_BUILD_PYTHON or create .venv and install the desktop dependencies first.');
process.exit(1);
