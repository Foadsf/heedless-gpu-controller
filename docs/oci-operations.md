# Field notes: operating the OCI controller

Measured on an Always Free `VM.Standard.E2.1.Micro` (AMD, 1 OCPU, 1 GB RAM)
running Ubuntu 22.04 **Minimal**, August 2026.

---

## Capacity: check before you plan

The most frustrating part of this setup is designing around a shape that turns
out to be unobtainable. OCI will answer directly, via the Compute Capacity
Report API (`POST /20160918/computeCapacityReports`), which returns
`AVAILABLE` / `OUT_OF_HOST_CAPACITY` / `HARDWARE_NOT_SUPPORTED` per shape.

Measured in `eu-amsterdam-1`:

| Shape | Free tier | Status |
|---|---|---|
| `VM.Standard.A1.Flex` (Ampere ARM) | yes | `OUT_OF_HOST_CAPACITY` at **every** size — 4/24, 2/12, 1/6, and the 1 OCPU / 1 GB minimum |
| `VM.Standard.E2.1.Micro` (AMD) | yes | `AVAILABLE` |

So a refused A1 is **not** a sizing problem you can shrink around: when a region
is out of Ampere, it is out at every size. Oracle also **reduced** the Always
Free A1 allowance to **2 OCPUs / 12 GB** (from 4/24), and some regions have a
single availability domain, so there is no second AD to retry in.

## Lockout recovery: Bastion, not Run Command

Most guides (including an earlier version of this repo's README) say to use OCI
**Run Command** to recover from a lost SSH key. On an Ubuntu **Minimal** image
with the snap-installed Oracle Cloud Agent, the
`Compute Instance Run Command` plugin is **absent from the plugin list
entirely** — and the console still lets you create a command, which then sits at
`Accepted` until it flips to `Expired`. That is a bad thing to discover on the
day you are locked out.

Check first: **Instance → Management → Oracle Cloud Agent**. If Run Command
isn't listed, use Bastion:

1. In that same plugin list, enable **Bastion** and wait for `Running`
   (a few minutes).
2. **Identity & Security → Bastion → Create bastion**, targeting the VCN and the
   instance's subnet. Restrict the CIDR allowlist to your own IP.
   *Bastion names are alphanumeric only — no hyphens.*
3. **Create session → Managed SSH**, username `ubuntu`, pick the instance, paste
   your **new** public key.
4. **View SSH command** gives you an `ssh -o ProxyCommand=…` line.

### The trap that will cost you the recovery twice

A Bastion managed session **appends its key to `~/.ssh/authorized_keys` and
removes that block when the session expires** (3 h max). If you append your
permanent key while a session is live, it lands adjacent to that block and
**goes away with it** — you get locked out again hours later, looking exactly
like a server-side revocation.

* Put permanent keys at **line 1**, with a distinguishing comment.
* Do not "verify" permanence while a bastion session is active — the injected
  key is usually the *same* key you're testing, so the two are indistinguishable.

## `Permission denied (publickey)` is a client-side claim

Twice this was not the server. The server's own log is the arbiter:

```sh
sudo journalctl -u ssh -g "Accepted|Failed" --no-pager | tail
```

If it shows `Accepted publickey` for your key and the failures read
`Connection closed by authenticating user … [preauth]`, then **your client hung
up** — no rejection happened. One cause is a desktop keyring's ssh-agent failing
to sign; the tell is `Offering public key … explicit agent` in `ssh -v`:

```
Host controller
    IdentityAgent none
    IdentitiesOnly yes
```

## Ubuntu Minimal has no rsyslog

`/var/log/auth.log` **does not exist**, so grepping it for intrusions returns a
confident, meaningless nothing. Use the journal:

```sh
sudo journalctl -u ssh -g "Accepted|Failed password|Invalid user" --no-pager
```

On a box with port 22 open to the internet, expect constant `Invalid user`
noise — one instance logged **673,465** attempts over 234 days. What matters is
whether any line says `Accepted` from an address that isn't yours. (In that
sample: none. All 18 successful logins came from the owner's IP and the
bastion's private endpoint.)

Also worth knowing: `grep -c "Accepted"` over the sshd journal **overcounts** —
it matches the substring inside `PubkeyAcceptedAlgorithms`, which appears in
every rejected-key line. Match `"Accepted publickey"`.

## Reboots and IPs

An **in-OS reboot** (`sudo systemctl reboot`) keeps an ephemeral public IP.
Console **Stop/Start** releases it. If you have automation pinned to the
address, prefer the former.

`ping` is **not** a liveness test: OCI's default security list allows TCP 22 but
not ICMP echo, so a perfectly healthy instance answers SSH and ignores ping. Use:

```sh
timeout 8 bash -c '</dev/tcp/YOUR_IP/22' && echo open || echo unreachable
```

## Idle reclamation

Oracle reclaims idle Always Free compute (7 days under 20% CPU/memory/network is
the commonly cited threshold). In practice an `E2.1.Micro` ran **234 days at
load 0.00** untouched, so the policy does not appear to be enforced against the
AMD micro shapes — but don't build on that if you ever obtain an A1.

## Talking to OCI from a script

There is a fully documented, officially supported REST API — the console itself
is a client of it. The endpoints are `<service>.<region>.oci.oraclecloud.com`,
and an unauthenticated call returns structured JSON:

```
GET https://iaas.<region>.oci.oraclecloud.com/20160918/instances
→ 401 {"code":"NotAuthenticated", ...}
```

Use API-key request signing (an RSA keypair added under *Profile → API keys*).
It does not expire, needs no browser, and works headless — unlike anything
built on a console session.

**Two probes that cannot fail here**, so they measure nothing:

* `404 NotAuthorizedOrNotFound` is Oracle's catch-all. A deliberately bogus
  path returns it *identically* to a real-but-forbidden one, so it cannot
  distinguish "no such endpoint" from "not authorised".
* Hostname probing hits a wildcard edge: `qqqxyzzy1zzz.<region>.oci.oraclecloud.com`
  answers exactly like a real service host. Never conclude a service exists
  because its hostname responded.
