# Commit message rules

Apply these rules when generating a Git commit message for AC-Prof.

- Generate one commit message for the entire diff, with exactly one subject.
  Do not generate a separate subject for each file or component.
- Write the subject and any body in English using `type(scope): description`,
  regardless of the editor language or the language of recent commits.
- Allowed types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `ci`, `build`, `chore`.
- Use `chore` for developer tooling configuration, `docs` for documentation-only
  changes, `test` for test-code changes, and `feat` for new runtime features.
- Use a short, lowercase scope naming the affected component, such as `profiling`,
  `energy`, `runtime`, `tui`, or `ci`. Omit the scope for repository-wide changes.
- Use imperative mood, keep the entire subject at most 72 characters, and omit
  the trailing period.
- Describe the concrete change. Avoid vague wording such as "improve functionality",
  "enhance code quality", or "optimize things".
- Base the message on the supplied diff. Do not infer changes from filenames alone
  or invent test results, validation, performance gains, or bug fixes.
- Prefer a single-line message. Add a body only when the diff supports details
  needed to explain the change. Leave a blank line after the subject, then use
  prose or bullets instead of additional `type(scope): description` subjects.
- Output only the commit message, without commentary or Markdown fences.

Example of one complete message:

```text
chore(git): configure commit message generation and checks
```
