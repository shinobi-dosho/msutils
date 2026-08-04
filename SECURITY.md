# Security Policy

## Supported versions

msutils applies security fixes to the latest release only; there are no
long-term-support branches.

| Version | Supported |
| ------- | --------- |
| latest `3.x` | :white_check_mark: |
| `2.x` and older | :x: |

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report vulnerabilities privately by email to **sphemakh@gmail.com** (or via
GitHub's [private vulnerability reporting][ghsa] on this repository, if
enabled). Include enough detail to reproduce — affected version, the operation
you ran, the Measurement Set's shape, and the impact you observed.

We aim to acknowledge reports within a reasonable time, work with you on a fix,
and credit you in the release notes if you'd like.

[ghsa]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

## Security posture

msutils reads and writes Measurement Sets on local disk. It has no network
surface, no daemon, and no credential handling. What it does have:

- **No `eval`/`exec`, ever.** Nothing msutils reads — not an MS, not a
  subtable, not a JSON summary — is executed. Metadata is parsed into the
  typed dataclasses in `msutils/info/_model.py` and nowhere else.
- **No shell.** msutils shells out to nothing. Every operation goes through
  python-casacore, so there is no `shell=True`, no string interpolation into a
  command line, and no `subprocess` in the library at all.
- **TaQL is executed, deliberately.** `msutils.taql()`, `msutils taql` and
  `subset(taql=...)` evaluate a TaQL expression against a table, which is the
  point of them. TaQL can read and write table data and, like SQL, can delete
  rows — so treat a TaQL string exactly as you would treat a SQL string:

  - Do not build one by concatenating untrusted input.
  - Do not accept one from a user you would not give write access to the MS.

  Where msutils builds TaQL itself (selections in `subset`, `flagstats`,
  `msinfo`), the values it interpolates are integer ids resolved through
  `MSInfo` registries, not raw user strings — a name that does not resolve
  raises before any query is built. Tables are bound positionally as `$1`
  rather than interpolated by path.
- **Writes are explicit and refuse to clobber.** `subset`, `average` and
  `convert` will not overwrite an existing output without `overwrite=True`;
  `flag_backup` will not replace a version without `overwrite=True`; `delcol`
  refuses to remove columns the MSv2 standard requires without `force=True`.
- **Untrusted Measurement Sets are still untrusted input.** An MS is a
  casacore table, and msutils hands it to python-casacore to parse. A
  malformed or hostile MS is a casacore parsing question, not something
  msutils can validate away — `msutils check` reports structural problems, but
  it is a conformance report, not a security boundary. Do not open MSs from
  sources you would not trust with the rest of your filesystem.

If you find a way around any of these, it's a security issue — please report
it as above.
