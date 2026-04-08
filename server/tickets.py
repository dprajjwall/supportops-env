"""
SupportOps-Env: Synthetic Ticket Dataset and Knowledge Base
60 realistic support tickets (ground-truth labels included).
20 KB articles for Task 3 (draft_response).
"""
from typing import List
from models import KBArticle, TicketData


# ─────────────────────────────────────────────
# Ticket Dataset
# ─────────────────────────────────────────────
TICKETS: List[TicketData] = [
    # ── BUG tickets ───────────────────────────
    TicketData(
        ticket_id="T001", subject="App crashes on login",
        body="Every time I try to log in from my iPhone the app crashes immediately. "
             "I have tried reinstalling but it doesn't help. This has been happening since yesterday.",
        customer_tier="enterprise", created_at="2024-01-15T09:00:00Z",
        sentiment_score=-0.9, sla_hours=4,
        category="Bug", priority="critical",
    ),
    TicketData(
        ticket_id="T002", subject="API returning 500 errors",
        body="Our production integration is getting HTTP 500 errors from /api/v2/orders. "
             "This is blocking our checkout flow. Started 2 hours ago. "
             "Error: Internal Server Error with no body.",
        customer_tier="enterprise", created_at="2024-01-15T10:30:00Z",
        sentiment_score=-0.95, sla_hours=4,
        category="Bug", priority="critical",
    ),
    TicketData(
        ticket_id="T003", subject="Dashboard charts not loading",
        body="The bar charts on the analytics dashboard are blank. "
             "Pie charts work fine. Chrome 120, macOS Sonoma.",
        customer_tier="pro", created_at="2024-01-15T11:00:00Z",
        sentiment_score=-0.5, sla_hours=24,
        category="Bug", priority="high",
    ),
    TicketData(
        ticket_id="T004", subject="Export to CSV missing columns",
        body="When exporting user data to CSV, the 'last_login' and 'created_at' columns "
             "are empty for all rows, even though the data shows in the UI.",
        customer_tier="pro", created_at="2024-01-16T08:00:00Z",
        sentiment_score=-0.4, sla_hours=24,
        category="Bug", priority="medium",
    ),
    TicketData(
        ticket_id="T005", subject="Typo on the pricing page",
        body="There is a small typo on /pricing: 'Unlimted' should be 'Unlimited'. "
             "Not a big deal but wanted to let you know!",
        customer_tier="free", created_at="2024-01-16T09:00:00Z",
        sentiment_score=0.3, sla_hours=72,
        category="Bug", priority="low",
    ),
    TicketData(
        ticket_id="T006", subject="Search autocomplete slow",
        body="The search dropdown takes 3-4 seconds to appear after typing. "
             "This is slightly annoying but doesn't block anything.",
        customer_tier="free", created_at="2024-01-16T10:00:00Z",
        sentiment_score=-0.2, sla_hours=72,
        category="Bug", priority="low",
    ),
    TicketData(
        ticket_id="T007", subject="Email notifications duplicated",
        body="I am receiving every notification email twice. "
             "This started after I updated my email address last week. "
             "Pro account, team of 5.",
        customer_tier="pro", created_at="2024-01-17T07:30:00Z",
        sentiment_score=-0.6, sla_hours=24,
        category="Bug", priority="high",
    ),
    TicketData(
        ticket_id="T008", subject="Mobile app freezes on settings",
        body="Opening 'Account Settings' on the Android app causes the app to freeze "
             "for about 10 seconds. App version 3.2.1, Android 14.",
        customer_tier="free", created_at="2024-01-17T11:00:00Z",
        sentiment_score=-0.3, sla_hours=72,
        category="Bug", priority="medium",
    ),
    TicketData(
        ticket_id="T009", subject="Webhook events not firing",
        body="Our webhook endpoint is not receiving any events. "
             "Verified the endpoint is live (returning 200). "
             "No events since 6 AM UTC today. Enterprise plan.",
        customer_tier="enterprise", created_at="2024-01-17T12:00:00Z",
        sentiment_score=-0.85, sla_hours=4,
        category="Bug", priority="critical",
    ),
    TicketData(
        ticket_id="T010", subject="Dark mode text contrast issue",
        body="In dark mode, some text labels in the sidebar appear very faint "
             "and are hard to read. Not a deal-breaker but accessibility concern.",
        customer_tier="free", created_at="2024-01-18T09:00:00Z",
        sentiment_score=0.0, sla_hours=72,
        category="Bug", priority="minimal",
    ),
    TicketData(
        ticket_id="T011", subject="Bulk delete not working",
        body="Selecting all items and clicking 'Delete Selected' does nothing. "
             "Individual delete works fine. Firefox 121.",
        customer_tier="pro", created_at="2024-01-18T10:30:00Z",
        sentiment_score=-0.5, sla_hours=24,
        category="Bug", priority="high",
    ),
    TicketData(
        ticket_id="T012", subject="SSO login loop",
        body="After configuring Okta SSO, users get stuck in a redirect loop. "
             "SAML assertion is correct according to Okta logs. "
             "All 200 users are blocked from logging in.",
        customer_tier="enterprise", created_at="2024-01-18T08:00:00Z",
        sentiment_score=-0.98, sla_hours=2,
        category="Bug", priority="critical",
    ),

    # ── FEATURE REQUEST tickets ───────────────
    TicketData(
        ticket_id="T013", subject="Add dark mode support",
        body="Would love a dark mode option in the settings. "
             "My eyes hurt using it at night. Many users want this too.",
        customer_tier="free", created_at="2024-01-15T12:00:00Z",
        sentiment_score=0.4, sla_hours=72,
        category="Feature Request", priority="low",
    ),
    TicketData(
        ticket_id="T014", subject="Request: Two-factor authentication",
        body="For security compliance we need 2FA support (TOTP preferred). "
             "This is a blocker for us moving to the enterprise tier.",
        customer_tier="pro", created_at="2024-01-15T13:00:00Z",
        sentiment_score=0.2, sla_hours=48,
        category="Feature Request", priority="high",
    ),
    TicketData(
        ticket_id="T015", subject="Bulk import via CSV",
        body="We have 5000 contacts to migrate. Currently we must add them one by one. "
             "A CSV import feature would save us many hours.",
        customer_tier="enterprise", created_at="2024-01-15T14:00:00Z",
        sentiment_score=0.1, sla_hours=24,
        category="Feature Request", priority="medium",
    ),
    TicketData(
        ticket_id="T016", subject="Slack integration",
        body="Is there a Slack integration or webhook for notifications? "
             "We would love to get alerts in our #ops channel.",
        customer_tier="pro", created_at="2024-01-16T09:30:00Z",
        sentiment_score=0.5, sla_hours=48,
        category="Feature Request", priority="medium",
    ),
    TicketData(
        ticket_id="T017", subject="Custom report builder",
        body="The predefined reports don't cover our needs. "
             "A drag-and-drop report builder with custom metrics would be very useful.",
        customer_tier="enterprise", created_at="2024-01-16T10:00:00Z",
        sentiment_score=0.3, sla_hours=24,
        category="Feature Request", priority="low",
    ),
    TicketData(
        ticket_id="T018", subject="API rate limit increase",
        body="Our current plan allows 1000 requests/minute but we need at least 5000. "
             "Can we get a higher limit or a custom plan?",
        customer_tier="enterprise", created_at="2024-01-16T11:00:00Z",
        sentiment_score=0.0, sla_hours=24,
        category="Feature Request", priority="medium",
    ),
    TicketData(
        ticket_id="T019", subject="Mobile app for Android",
        body="Do you have plans for an Android app? The iOS app is great "
             "but half our team uses Android.",
        customer_tier="free", created_at="2024-01-17T08:00:00Z",
        sentiment_score=0.2, sla_hours=72,
        category="Feature Request", priority="minimal",
    ),
    TicketData(
        ticket_id="T020", subject="Audit log for compliance",
        body="We need a full audit log of all user actions for SOC2 compliance. "
             "Who changed what settings and when. This is required for our certification.",
        customer_tier="enterprise", created_at="2024-01-17T09:00:00Z",
        sentiment_score=0.0, sla_hours=24,
        category="Feature Request", priority="high",
    ),
    TicketData(
        ticket_id="T021", subject="Keyboard shortcuts",
        body="Would be great to have keyboard shortcuts for common actions. "
             "Power users would love this for efficiency.",
        customer_tier="free", created_at="2024-01-18T08:00:00Z",
        sentiment_score=0.4, sla_hours=72,
        category="Feature Request", priority="minimal",
    ),
    TicketData(
        ticket_id="T022", subject="Zapier integration request",
        body="Would love to connect your app to our other tools via Zapier. "
             "This would automate a lot of manual work.",
        customer_tier="pro", created_at="2024-01-18T09:00:00Z",
        sentiment_score=0.3, sla_hours=48,
        category="Feature Request", priority="low",
    ),
    TicketData(
        ticket_id="T023", subject="White-labeling option",
        body="We resell to clients and need the ability to white-label the interface "
             "with our brand colors and logo. Happy to pay extra.",
        customer_tier="enterprise", created_at="2024-01-18T10:00:00Z",
        sentiment_score=0.2, sla_hours=24,
        category="Feature Request", priority="medium",
    ),
    TicketData(
        ticket_id="T024", subject="Multi-language support",
        body="Our users are in France and Germany. French and German UI support "
             "would help us expand.",
        customer_tier="pro", created_at="2024-01-19T08:00:00Z",
        sentiment_score=0.3, sla_hours=48,
        category="Feature Request", priority="low",
    ),

    # ── BILLING tickets ───────────────────────
    TicketData(
        ticket_id="T025", subject="Charged twice this month",
        body="I see two charges of $149 on my credit card for January. "
             "Order IDs: inv_8821 and inv_8930. Please refund the duplicate.",
        customer_tier="pro", created_at="2024-01-15T15:00:00Z",
        sentiment_score=-0.8, sla_hours=12,
        category="Billing", priority="high",
    ),
    TicketData(
        ticket_id="T026", subject="Invoice not received",
        body="I upgraded to the Pro plan 3 days ago but still haven't received "
             "the invoice by email. I need it for accounting.",
        customer_tier="pro", created_at="2024-01-16T08:00:00Z",
        sentiment_score=-0.3, sla_hours=24,
        category="Billing", priority="medium",
    ),
    TicketData(
        ticket_id="T027", subject="Need to update payment card",
        body="My card expires next month. How do I update the payment method? "
             "I can't find the option in the settings.",
        customer_tier="free", created_at="2024-01-16T09:00:00Z",
        sentiment_score=0.0, sla_hours=72,
        category="Billing", priority="low",
    ),
    TicketData(
        ticket_id="T028", subject="Annual plan discount query",
        body="I am on monthly Pro at $149/mo. If I switch to annual, what is the price? "
             "Do you offer any additional discounts for 3-year commitments?",
        customer_tier="pro", created_at="2024-01-16T10:00:00Z",
        sentiment_score=0.3, sla_hours=48,
        category="Billing", priority="low",
    ),
    TicketData(
        ticket_id="T029", subject="VAT not applied on invoice",
        body="My company is EU-based (Germany). The invoice does not show VAT / "
             "tax ID which is required by law. Please reissue with VAT details.",
        customer_tier="enterprise", created_at="2024-01-17T08:30:00Z",
        sentiment_score=-0.4, sla_hours=24,
        category="Billing", priority="medium",
    ),
    TicketData(
        ticket_id="T030", subject="Refund request after cancellation",
        body="I cancelled my subscription yesterday (within 3 days of renewal). "
             "Per your 30-day refund policy, I should be eligible. "
             "Please process the $299 refund.",
        customer_tier="pro", created_at="2024-01-17T09:00:00Z",
        sentiment_score=-0.7, sla_hours=24,
        category="Billing", priority="high",
    ),
    TicketData(
        ticket_id="T031", subject="Upgrade to Enterprise pricing",
        body="We have 50 users and need Enterprise features. "
             "Can you send a custom quote? Revenue around $5M/year.",
        customer_tier="pro", created_at="2024-01-17T10:00:00Z",
        sentiment_score=0.2, sla_hours=24,
        category="Billing", priority="medium",
    ),
    TicketData(
        ticket_id="T032", subject="Payment failure - urgent",
        body="Our subscription payment failed but we cannot update our card "
             "because the settings page shows an error. "
             "We have 300 users that cannot access the platform.",
        customer_tier="enterprise", created_at="2024-01-17T11:00:00Z",
        sentiment_score=-0.95, sla_hours=4,
        category="Billing", priority="critical",
    ),
    TicketData(
        ticket_id="T033", subject="Billing address change",
        body="We moved offices. Can you update our billing address on future invoices? "
             "New address: 123 Tech Street, San Francisco, CA 94105.",
        customer_tier="pro", created_at="2024-01-18T08:00:00Z",
        sentiment_score=0.1, sla_hours=48,
        category="Billing", priority="low",
    ),
    TicketData(
        ticket_id="T034", subject="Nonprofit discount available?",
        body="We are a registered 501(c)(3) nonprofit. Do you offer any special "
             "pricing for nonprofits? We cannot afford the full Pro price.",
        customer_tier="free", created_at="2024-01-18T09:00:00Z",
        sentiment_score=0.0, sla_hours=72,
        category="Billing", priority="minimal",
    ),
    TicketData(
        ticket_id="T035", subject="Subscription auto-renewed unexpectedly",
        body="My annual subscription auto-renewed and charged $1788 without warning. "
             "I did not see any renewal reminder. I want a full refund.",
        customer_tier="enterprise", created_at="2024-01-18T10:00:00Z",
        sentiment_score=-0.9, sla_hours=8,
        category="Billing", priority="critical",
    ),
    TicketData(
        ticket_id="T036", subject="Need itemized invoice breakdown",
        body="Our finance team needs a detailed breakdown of the enterprise invoice "
             "showing cost per seat, features included, and overage charges.",
        customer_tier="enterprise", created_at="2024-01-19T08:00:00Z",
        sentiment_score=-0.1, sla_hours=24,
        category="Billing", priority="medium",
    ),

    # ── ACCOUNT tickets ───────────────────────
    TicketData(
        ticket_id="T037", subject="Cannot reset password",
        body="I requested a password reset but the email never arrived. "
             "I have checked spam. Been locked out for 2 hours.",
        customer_tier="pro", created_at="2024-01-15T16:00:00Z",
        sentiment_score=-0.7, sla_hours=12,
        category="Account", priority="high",
    ),
    TicketData(
        ticket_id="T038", subject="Delete my account and all data",
        body="Please permanently delete my account and all associated data per GDPR. "
             "Account email: john.doe@example.com",
        customer_tier="free", created_at="2024-01-16T08:00:00Z",
        sentiment_score=-0.2, sla_hours=72,
        category="Account", priority="medium",
    ),
    TicketData(
        ticket_id="T039", subject="Transfer account ownership",
        body="I am leaving the company and need to transfer the account to my colleague "
             "Sarah Jones (sarah@acme.com). She should become the new admin.",
        customer_tier="enterprise", created_at="2024-01-16T09:00:00Z",
        sentiment_score=0.0, sla_hours=24,
        category="Account", priority="medium",
    ),
    TicketData(
        ticket_id="T040", subject="Add team member stuck",
        body="When I try to invite a new team member via Settings > Team, "
             "I click 'Send Invite' but nothing happens. No email is sent.",
        customer_tier="pro", created_at="2024-01-16T10:00:00Z",
        sentiment_score=-0.4, sla_hours=24,
        category="Account", priority="medium",
    ),
    TicketData(
        ticket_id="T041", subject="How do I change my email address?",
        body="I want to update the email on my account from old@email.com to new@email.com. "
             "Where is this option? I can't find it.",
        customer_tier="free", created_at="2024-01-17T08:00:00Z",
        sentiment_score=0.0, sla_hours=72,
        category="Account", priority="low",
    ),
    TicketData(
        ticket_id="T042", subject="Suspicious login from unknown location",
        body="I received an alert about a login from Russia, but I am in the US. "
             "Please lock my account immediately and help me secure it.",
        customer_tier="enterprise", created_at="2024-01-17T07:00:00Z",
        sentiment_score=-0.95, sla_hours=2,
        category="Account", priority="critical",
    ),
    TicketData(
        ticket_id="T043", subject="Set up SSO for our company",
        body="We want to configure SAML SSO with our internal IdP. "
             "Where can I find the SAML metadata URL and instructions?",
        customer_tier="enterprise", created_at="2024-01-17T10:00:00Z",
        sentiment_score=0.1, sla_hours=24,
        category="Account", priority="medium",
    ),
    TicketData(
        ticket_id="T044", subject="Profile picture not updating",
        body="I uploaded a new profile picture but the old one still shows everywhere. "
             "Tried clearing cache. Account: user123.",
        customer_tier="free", created_at="2024-01-18T09:00:00Z",
        sentiment_score=-0.1, sla_hours=72,
        category="Account", priority="minimal",
    ),
    TicketData(
        ticket_id="T045", subject="Remove user from organization",
        body="An employee left the company. Can you help me remove them from our "
             "organization and revoke all access? User: alice@company.com",
        customer_tier="pro", created_at="2024-01-18T10:00:00Z",
        sentiment_score=0.0, sla_hours=24,
        category="Account", priority="high",
    ),
    TicketData(
        ticket_id="T046", subject="Username already taken",
        body="I want to use the username 'techcorp' but it says it's taken. "
             "The account has been inactive for years. Can you release it?",
        customer_tier="free", created_at="2024-01-19T08:00:00Z",
        sentiment_score=-0.1, sla_hours=72,
        category="Account", priority="minimal",
    ),
    TicketData(
        ticket_id="T047", subject="Need data export for compliance audit",
        body="Our compliance audit requires a full export of all data, logs and "
             "configurations for the last 12 months. Enterprise contract requires "
             "this within 24 hours.",
        customer_tier="enterprise", created_at="2024-01-19T09:00:00Z",
        sentiment_score=-0.3, sla_hours=24,
        category="Account", priority="high",
    ),
    TicketData(
        ticket_id="T048", subject="Two admins locked out after 2FA reset",
        body="Both our admins lost access to their 2FA apps simultaneously "
             "(new phones). We are completely locked out of the organization.",
        customer_tier="enterprise", created_at="2024-01-19T06:00:00Z",
        sentiment_score=-0.99, sla_hours=2,
        category="Account", priority="critical",
    ),

    # ── GENERAL tickets ───────────────────────
    TicketData(
        ticket_id="T049", subject="How do I get started?",
        body="I just signed up and I'm not sure where to begin. "
             "Is there a getting started guide or tutorial?",
        customer_tier="free", created_at="2024-01-15T17:00:00Z",
        sentiment_score=0.4, sla_hours=72,
        category="General", priority="minimal",
    ),
    TicketData(
        ticket_id="T050", subject="What are the system requirements?",
        body="We want to deploy on-premise. What are the hardware and OS requirements? "
             "We have Ubuntu 22.04 servers.",
        customer_tier="enterprise", created_at="2024-01-16T08:00:00Z",
        sentiment_score=0.2, sla_hours=24,
        category="General", priority="low",
    ),
    TicketData(
        ticket_id="T051", subject="When is the next maintenance window?",
        body="We have a critical demo on Friday. Is any maintenance scheduled "
             "this week that could affect availability?",
        customer_tier="enterprise", created_at="2024-01-16T09:00:00Z",
        sentiment_score=0.0, sla_hours=24,
        category="General", priority="medium",
    ),
    TicketData(
        ticket_id="T052", subject="Do you offer training for teams?",
        body="We are onboarding 30 people next month. Do you offer group training "
             "sessions or webinars? Happy to pay for a premium onboarding session.",
        customer_tier="enterprise", created_at="2024-01-16T10:00:00Z",
        sentiment_score=0.4, sla_hours=24,
        category="General", priority="low",
    ),
    TicketData(
        ticket_id="T053", subject="GDPR data processing agreement",
        body="We are EU-based and need to sign a Data Processing Agreement (DPA) "
             "before we can use your service. Can you send the DPA for signature?",
        customer_tier="enterprise", created_at="2024-01-17T08:00:00Z",
        sentiment_score=0.0, sla_hours=24,
        category="General", priority="medium",
    ),
    TicketData(
        ticket_id="T054", subject="Is there a status page?",
        body="Where can I check service status and incident history? "
             "Do you have a status page I can bookmark?",
        customer_tier="free", created_at="2024-01-17T09:00:00Z",
        sentiment_score=0.1, sla_hours=72,
        category="General", priority="minimal",
    ),
    TicketData(
        ticket_id="T055", subject="API documentation unclear",
        body="The docs for the /events endpoint are confusing. "
             "The example request doesn't match the schema shown above it.",
        customer_tier="pro", created_at="2024-01-17T10:00:00Z",
        sentiment_score=-0.3, sla_hours=48,
        category="General", priority="low",
    ),
    TicketData(
        ticket_id="T056", subject="Partnership inquiry",
        body="We are a consulting firm and want to explore a reseller partnership. "
             "Who should I talk to about partnership opportunities?",
        customer_tier="free", created_at="2024-01-18T08:00:00Z",
        sentiment_score=0.5, sla_hours=72,
        category="General", priority="minimal",
    ),
    TicketData(
        ticket_id="T057", subject="Uptime SLA details",
        body="Our enterprise contract mentions 99.9% uptime SLA. "
             "How is downtime measured and how do we claim credits?",
        customer_tier="enterprise", created_at="2024-01-18T09:00:00Z",
        sentiment_score=0.1, sla_hours=24,
        category="General", priority="low",
    ),
    TicketData(
        ticket_id="T058", subject="Data retention policy",
        body="How long do you retain user data after account deletion? "
             "We need this info for our privacy policy.",
        customer_tier="pro", created_at="2024-01-18T10:00:00Z",
        sentiment_score=0.0, sla_hours=48,
        category="General", priority="low",
    ),
    TicketData(
        ticket_id="T059", subject="Feature roadmap available?",
        body="Is there a public roadmap where we can see upcoming features? "
             "We are evaluating whether to commit long-term.",
        customer_tier="pro", created_at="2024-01-19T08:00:00Z",
        sentiment_score=0.2, sla_hours=48,
        category="General", priority="minimal",
    ),
    TicketData(
        ticket_id="T060", subject="Integration with Salesforce broken",
        body="Our Salesforce sync stopped working 30 minutes ago. "
             "No CRM records are being pushed. This affects our entire sales team of 80 people. "
             "Enterprise deal at risk.",
        customer_tier="enterprise", created_at="2024-01-19T07:00:00Z",
        sentiment_score=-0.9, sla_hours=4,
        category="Bug", priority="critical",
    ),
]

# Build an index for easy access
TICKET_INDEX = {t.ticket_id: t for t in TICKETS}

# ─────────────────────────────────────────────
# Pre-defined Priority Sets for Task 2
# Each set has 5 tickets; ground_truth_order is 1=most urgent, 5=least urgent
# ─────────────────────────────────────────────
PRIORITY_SETS = [
    {
        "set_id": "PS001",
        "ticket_ids": ["T012", "T037", "T026", "T052", "T054"],
        "ground_truth_order": ["T012", "T037", "T026", "T052", "T054"],
        "rationale": "SSO outage > password lockout > missing invoice > training > FAQ",
    },
    {
        "set_id": "PS002",
        "ticket_ids": ["T002", "T009", "T007", "T004", "T005"],
        "ground_truth_order": ["T002", "T009", "T007", "T004", "T005"],
        "rationale": "Prod API 500 > webhook failure > email dups > CSV bug > typo",
    },
    {
        "set_id": "PS003",
        "ticket_ids": ["T048", "T042", "T032", "T025", "T028"],
        "ground_truth_order": ["T048", "T042", "T032", "T025", "T028"],
        "rationale": "Org lockout > security breach > payment fail > duplicate charge > discount query",
    },
    {
        "set_id": "PS004",
        "ticket_ids": ["T060", "T001", "T011", "T016", "T059"],
        "ground_truth_order": ["T060", "T001", "T011", "T016", "T059"],
        "rationale": "Salesforce sync > app crash > bulk delete > slack integration > roadmap",
    },
    {
        "set_id": "PS005",
        "ticket_ids": ["T035", "T030", "T029", "T033", "T034"],
        "ground_truth_order": ["T035", "T030", "T029", "T033", "T034"],
        "rationale": "Auto-renewal shock > refund request > VAT issue > address change > nonprofit",
    },
]

# ─────────────────────────────────────────────
# Draft-response tasks (Task 3)
# Each task has 1 ticket + expected KB topics
# ─────────────────────────────────────────────
DRAFT_TASKS = [
    {
        "task_id": "DT001",
        "ticket_id": "T037",  # Cannot reset password
        "required_kb_tags": ["password-reset", "account-access"],
        "expected_resolution_keywords": ["password", "reset", "email", "spam", "link"],
        "expected_tone_words": ["apologize", "help", "sorry", "assist", "understand"],
    },
    {
        "task_id": "DT002",
        "ticket_id": "T002",  # API 500 errors
        "required_kb_tags": ["api", "errors", "troubleshooting"],
        "expected_resolution_keywords": ["investigating", "status", "team", "escalate", "engineers"],
        "expected_tone_words": ["apologize", "impact", "priority", "urgency", "team"],
    },
    {
        "task_id": "DT003",
        "ticket_id": "T025",  # Charged twice
        "required_kb_tags": ["billing", "refund", "payment"],
        "expected_resolution_keywords": ["refund", "duplicate", "charge", "process", "credit"],
        "expected_tone_words": ["apologize", "inconvenience", "immediately", "process", "understand"],
    },
    {
        "task_id": "DT004",
        "ticket_id": "T039",  # Transfer account ownership
        "required_kb_tags": ["account", "ownership", "team-management"],
        "expected_resolution_keywords": ["transfer", "ownership", "admin", "verify", "email"],
        "expected_tone_words": ["help", "assist", "process", "confirm", "verify"],
    },
    {
        "task_id": "DT005",
        "ticket_id": "T053",  # GDPR DPA
        "required_kb_tags": ["gdpr", "compliance", "legal"],
        "expected_resolution_keywords": ["DPA", "agreement", "sign", "GDPR", "legal"],
        "expected_tone_words": ["certainly", "send", "provide", "compliance", "team"],
    },
]

# ─────────────────────────────────────────────
# Knowledge Base Articles
# ─────────────────────────────────────────────
KB_ARTICLES: List[KBArticle] = [
    KBArticle(
        article_id="KB001",
        title="How to Reset Your Password",
        content=(
            "To reset your password: 1) Click 'Forgot Password' on the login page. "
            "2) Enter your registered email address. "
            "3) Check your inbox for a reset link (also check spam/junk folder). "
            "4) Click the link within 24 hours — it expires after that. "
            "5) Choose a new password that is at least 8 characters. "
            "If you don't receive the email, check your spam folder or contact support."
        ),
        tags=["password-reset", "account-access", "login"],
    ),
    KBArticle(
        article_id="KB002",
        title="API Troubleshooting Guide",
        content=(
            "Common API issues and solutions: "
            "500 Internal Server Error: This indicates a server-side problem. "
            "Check our status page at status.example.com for ongoing incidents. "
            "If you see 500 errors, please share the request ID from the X-Request-Id header. "
            "Rate limit errors (429): You have exceeded your plan's rate limit. "
            "Upgrade your plan or implement exponential backoff. "
            "Authentication errors (401/403): Verify your API key is correct and not expired."
        ),
        tags=["api", "errors", "troubleshooting", "rate-limit"],
    ),
    KBArticle(
        article_id="KB003",
        title="Refund and Billing Policy",
        content=(
            "Our refund policy: We offer a 30-day money-back guarantee for all plans. "
            "To request a refund, contact support with your order ID. "
            "Duplicate charges: If you've been charged twice, please provide both invoice numbers "
            "and we will immediately process a refund for the duplicate. "
            "Refunds typically appear within 5-10 business days on your statement. "
            "For annual plans, refunds are prorated after 30 days."
        ),
        tags=["billing", "refund", "payment", "policy"],
    ),
    KBArticle(
        article_id="KB004",
        title="Account Ownership Transfer",
        content=(
            "To transfer account ownership: "
            "1) The current owner must initiate the transfer from Settings > Team > Transfer Ownership. "
            "2) Enter the email of the new owner. "
            "3) The new owner will receive a confirmation email and must accept within 72 hours. "
            "4) Once accepted, the previous owner's role changes to Admin. "
            "For enterprise accounts requiring special handling, contact support with both user emails "
            "and we will process the transfer manually after identity verification."
        ),
        tags=["account", "ownership", "team-management", "admin"],
    ),
    KBArticle(
        article_id="KB005",
        title="GDPR Data Processing Agreement",
        content=(
            "We are GDPR compliant and provide a Data Processing Agreement (DPA) for all enterprise customers. "
            "To request a DPA: Email legal@example.com with your company name and DPA request. "
            "We will send a pre-signed DPA within 2 business days. "
            "Our DPA covers: data processing purposes, data categories, retention periods, "
            "sub-processors, and your rights under GDPR Articles 13-17. "
            "Standard Contractual Clauses (SCCs) are included for non-EU data transfers."
        ),
        tags=["gdpr", "compliance", "legal", "dpa", "privacy"],
    ),
    KBArticle(
        article_id="KB006",
        title="Two-Factor Authentication Setup",
        content=(
            "Enable 2FA from Settings > Security > Two-Factor Authentication. "
            "We support TOTP authenticator apps (Google Authenticator, Authy). "
            "Lost your 2FA device? Contact support with your account email and government ID. "
            "For enterprise accounts with organization-wide 2FA lockout, the account owner "
            "must contact support with written authorization from 2 senior employees."
        ),
        tags=["2fa", "security", "account-access", "authentication"],
    ),
    KBArticle(
        article_id="KB007",
        title="SSO and SAML Configuration",
        content=(
            "Setting up SAML SSO: "
            "1) Go to Settings > Security > Single Sign-On. "
            "2) Download our SAML metadata XML. "
            "3) Configure your IdP (Okta, Azure AD, Google Workspace) with our metadata. "
            "4) Enter your IdP metadata URL in our settings. "
            "Common issues: Attribute mapping errors (ensure email, firstName, lastName are mapped). "
            "Redirect loops: Check that your IdP callback URL matches exactly: "
            "https://app.example.com/auth/saml/callback"
        ),
        tags=["sso", "saml", "security", "enterprise", "authentication"],
    ),
    KBArticle(
        article_id="KB008",
        title="Webhook Configuration and Troubleshooting",
        content=(
            "Webhooks allow real-time event notifications to your server. "
            "Setup: Settings > Integrations > Webhooks > Add Webhook URL. "
            "Your endpoint must return HTTP 200 within 5 seconds. "
            "Not receiving events? Check: 1) Endpoint is publicly accessible, "
            "2) Returns 200 status code, 3) No IP allowlisting blocking our IPs. "
            "Our webhook IPs: 54.23.x.x range. See docs for full IP list. "
            "Failed deliveries are retried 3 times with exponential backoff."
        ),
        tags=["webhook", "integration", "api", "troubleshooting"],
    ),
    KBArticle(
        article_id="KB009",
        title="Pricing Plans and Upgrades",
        content=(
            "Plans: Free (up to 3 users), Pro ($149/mo, up to 50 users), "
            "Enterprise (custom pricing, unlimited users). "
            "Annual discount: 20% off for annual commitment. "
            "Multi-year deals: Contact sales for 2+ year pricing (up to 35% discount). "
            "Nonprofit pricing: 30% discount available for verified 501(c)(3) organizations. "
            "To upgrade: Settings > Billing > Change Plan. "
            "For enterprise quotes, contact sales@example.com."
        ),
        tags=["pricing", "plans", "upgrade", "billing"],
    ),
    KBArticle(
        article_id="KB010",
        title="CSV Import and Data Migration",
        content=(
            "Bulk import is available on Pro and Enterprise plans. "
            "Settings > Data > Import > Upload CSV. "
            "Supported formats: CSV, XLSX (max 10,000 rows per file). "
            "Required columns: name, email. Optional: phone, company, tags. "
            "After upload, preview changes before confirming the import. "
            "For imports over 10,000 records, contact support for assisted migration."
        ),
        tags=["import", "csv", "migration", "data"],
    ),
    KBArticle(
        article_id="KB011",
        title="Integrations: Slack, Zapier, and Salesforce",
        content=(
            "Slack: Settings > Integrations > Slack > Connect. "
            "Select channels for different notification types. "
            "Zapier: Create a Zap using our official Zapier app (search 'SupportOps'). "
            "Salesforce: Settings > Integrations > Salesforce. "
            "If Salesforce sync breaks, try: 1) Disconnect and reconnect, "
            "2) Check OAuth token hasn't expired, 3) Verify field mapping hasn't changed."
        ),
        tags=["slack", "zapier", "salesforce", "integration"],
    ),
    KBArticle(
        article_id="KB012",
        title="Removing a Team Member",
        content=(
            "To remove a user from your organization: "
            "Settings > Team > Find the user > Click '...' > Remove. "
            "The user loses all access immediately. "
            "Their data and contributions remain in the system. "
            "If the user you're removing is the account owner, "
            "first transfer ownership to another admin. "
            "Deactivated users do not count toward your user limit within 24 hours."
        ),
        tags=["account", "team-management", "user-management"],
    ),
    KBArticle(
        article_id="KB013",
        title="Data Export and Compliance Reports",
        content=(
            "Export your data: Settings > Data > Export. "
            "Available formats: CSV, JSON, XML. "
            "For compliance audits (SOC2, ISO 27001, GDPR), we provide: "
            "- Audit logs: Settings > Audit Log (Enterprise only) "
            "- Full account export: Settings > Data > Full Export "
            "- Data processing records on request through support. "
            "Exports for the past 12 months are available for Enterprise customers."
        ),
        tags=["export", "compliance", "audit", "gdpr", "data"],
    ),
    KBArticle(
        article_id="KB014",
        title="Service Status and Maintenance Windows",
        content=(
            "Check real-time service status: status.example.com "
            "Subscribe to status updates via email or RSS. "
            "Scheduled maintenance: We aim to schedule maintenance during low-traffic windows "
            "(Sundays 2–6 AM UTC). Advance notice is 72+ hours via email and status page. "
            "SLA Credits: If uptime drops below 99.9% in a month, Enterprise customers "
            "receive a 10% credit on their monthly bill. Claim via support within 30 days."
        ),
        tags=["status", "maintenance", "uptime", "sla"],
    ),
    KBArticle(
        article_id="KB015",
        title="Password and Account Security Best Practices",
        content=(
            "Use a strong, unique password (minimum 12 characters with mixed case, numbers, symbols). "
            "Enable Two-Factor Authentication (Settings > Security > 2FA). "
            "Suspicious login alerts: If you receive an alert for a login you don't recognize, "
            "immediately change your password and enable 2FA. "
            "Contact support to review active sessions and revoke any unauthorized access. "
            "We will freeze the account if there is evidence of compromise."
        ),
        tags=["security", "password", "account-access", "2fa"],
    ),
    KBArticle(
        article_id="KB016",
        title="Email Notification Settings",
        content=(
            "Manage email notifications: Settings > Notifications > Email Preferences. "
            "If you're receiving duplicate notifications: "
            "1) Check if you have multiple accounts with the same email. "
            "2) Verify notification rules are not duplicated (Settings > Automation). "
            "3) Try removing and re-adding your email address in account settings. "
            "Notifications not arriving: Check your email provider's spam filter "
            "and whitelist notifications@example.com."
        ),
        tags=["notifications", "email", "settings"],
    ),
    KBArticle(
        article_id="KB017",
        title="Getting Started Guide",
        content=(
            "Welcome to SupportOps! Quick start: "
            "1) Invite your team: Settings > Team > Invite Members. "
            "2) Connect your inbox: Settings > Integrations > Email. "
            "3) Set up ticket categories: Settings > Configuration > Categories. "
            "4) Watch our 5-minute onboarding video at docs.example.com/quickstart. "
            "5) Book a free onboarding call with our CS team: calendly.com/supportops-onboarding. "
            "Full documentation: docs.example.com"
        ),
        tags=["onboarding", "getting-started", "tutorial"],
    ),
    KBArticle(
        article_id="KB018",
        title="Billing: Updating Payment Method",
        content=(
            "Update your payment method: Settings > Billing > Payment Methods > Add Card. "
            "We accept Visa, Mastercard, Amex, and ACH (Enterprise). "
            "If Settings > Billing shows an error, try: "
            "1) Clear browser cache and retry. "
            "2) Use an incognito window. "
            "3) Contact support — we can securely update your card via phone verification. "
            "Payment failures: We retry failed payments 3 times over 7 days before suspending the account."
        ),
        tags=["billing", "payment", "card", "subscription"],
    ),
    KBArticle(
        article_id="KB019",
        title="API Rate Limits",
        content=(
            "Rate limits by plan: Free: 100 req/min, Pro: 1,000 req/min, Enterprise: custom. "
            "Headers included in every response: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. "
            "When you hit the limit, you receive a 429 Too Many Requests response. "
            "Best practice: implement exponential backoff starting at 1 second. "
            "For Enterprise customers needing higher limits, contact sales@example.com "
            "for a custom rate limit agreement."
        ),
        tags=["api", "rate-limit", "enterprise", "performance"],
    ),
    KBArticle(
        article_id="KB020",
        title="Data Retention and Deletion Policy",
        content=(
            "Active accounts: Data is retained indefinitely while your account is active. "
            "After account deletion: User-identifiable data is deleted within 30 days. "
            "Backup retention: Encrypted backups are purged within 90 days. "
            "Audit logs: Retained for 12 months (Enterprise) or 3 months (others). "
            "GDPR Right to Erasure: Contact privacy@example.com with your request. "
            "We confirm deletion within 30 days as required by GDPR Article 17."
        ),
        tags=["gdpr", "privacy", "data-retention", "compliance", "deletion"],
    ),
]

# Build KB index by article_id and by tag
KB_INDEX = {a.article_id: a for a in KB_ARTICLES}
KB_TAG_INDEX: dict[str, list[KBArticle]] = {}
for article in KB_ARTICLES:
    for tag in article.tags:
        KB_TAG_INDEX.setdefault(tag, []).append(article)


def search_kb(query: str, max_results: int = 3) -> list[str]:
    """
    Simple keyword-based KB search.
    Returns list of formatted article strings.
    """
    query_lower = query.lower()
    scores: dict[str, float] = {}

    for article in KB_ARTICLES:
        score = 0.0
        text = f"{article.title} {article.content} {' '.join(article.tags)}".lower()
        for word in query_lower.split():
            if len(word) >= 3 and word in text:
                score += 1.0
                # Bonus if word matches title or tag
                if word in article.title.lower():
                    score += 0.5
                if any(word in tag for tag in article.tags):
                    score += 0.3
        if score > 0:
            scores[article.article_id] = score

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
    results = []
    for article_id, _ in top:
        article = KB_INDEX[article_id]
        results.append(
            f"[{article.article_id}] {article.title}\n{article.content}"
        )
    return results if results else ["No articles found matching your query."]
