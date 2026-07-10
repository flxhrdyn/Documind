import { describe, it, expect } from 'vitest';
import { nextPollInterval } from './useUploadJob';

describe('nextPollInterval', () => {
  it('doubles then caps at 5000ms', () => {
    expect(nextPollInterval(1000)).toBe(2000);
    expect(nextPollInterval(2000)).toBe(4000);
    expect(nextPollInterval(4000)).toBe(5000);
    expect(nextPollInterval(5000)).toBe(5000);
  });
});
