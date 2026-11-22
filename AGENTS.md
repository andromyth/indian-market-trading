# Repository Agent Instructions

## Git commit identity (required)

Before creating any commit or pushing changes in this repository, set and verify the repository-local Git identity:

```bash
git config --local user.name "andromyth"
git config --local user.email "arindamsil@duck.com"
git config --local --get user.name
git config --local --get user.email
```

The expected output is exactly:

- `andromyth`
- `arindamsil@duck.com`

Repository-local configuration is intentional: it overrides any global identity supplied by an agent runtime. Do not use `OpenCode`, `opencode@users.noreply.github.com`, or another agent/runtime identity as author or committer.

Before pushing, inspect the pending commit metadata:

```bash
git log -1 --format='author=%an <%ae>%ncommitter=%cn <%ce>'
```

Do not push if either identity differs from the expected values. Correct the local configuration and amend only the not-yet-published commit if necessary. Never rewrite already-published history without explicit user approval.
