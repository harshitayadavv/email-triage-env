from __future__ import annotations
from typing import List
from env.models import Email


# ── 20 synthetic emails with ground truth labels and priorities ─────

ALL_EMAILS = [

    # --- URGENT (priority 1-4) ---
    Email(
        id="e1",
        sender="cto@company.com",
        subject="URGENT: Production API is down",
        body="Our main API has been returning 500 errors for the last 30 minutes. Customers are complaining. We need this fixed immediately. All hands on deck.",
        timestamp="2024-01-15T09:15:00Z",
        true_label="urgent",
        true_priority=1,
    ),
    Email(
        id="e2",
        sender="security@company.com",
        subject="CRITICAL: Suspected data breach detected",
        body="Our monitoring system has flagged unusual data access patterns on the production database. Possible unauthorized access. Please review immediately and escalate to the security team.",
        timestamp="2024-01-15T08:45:00Z",
        true_label="urgent",
        true_priority=2,
    ),
    Email(
        id="e3",
        sender="client.vip@bigcorp.com",
        subject="System outage affecting our business",
        body="We have been unable to access your platform for the past hour. This is directly impacting our revenue. We need an immediate response and resolution timeline or we will have to escalate to your CEO.",
        timestamp="2024-01-15T09:30:00Z",
        true_label="urgent",
        true_priority=3,
    ),
    Email(
        id="e4",
        sender="hr@company.com",
        subject="Action required: Complete compliance training by EOD",
        body="This is your final reminder. All employees must complete the mandatory compliance training by end of business today. Failure to do so may result in account suspension.",
        timestamp="2024-01-15T07:00:00Z",
        true_label="urgent",
        true_priority=4,
    ),

    # --- ACTION-NEEDED (priority 5-8) ---
    Email(
        id="e5",
        sender="finance@company.com",
        subject="Q4 budget approval needed before Friday",
        body="Hi, the Q4 budget proposal is ready for your review. We need your sign-off before Friday so procurement can proceed. Attached is the full breakdown. Please approve or send back with comments.",
        timestamp="2024-01-15T10:00:00Z",
        true_label="action-needed",
        true_priority=5,
    ),
    Email(
        id="e6",
        sender="recruiter@company.com",
        subject="Interview scheduling: Senior Engineer candidate",
        body="We have a strong candidate for the Senior Engineer role. They are available Tuesday or Wednesday next week. Could you confirm your availability for a 45-minute panel interview?",
        timestamp="2024-01-15T11:00:00Z",
        true_label="action-needed",
        true_priority=6,
    ),
    Email(
        id="e7",
        sender="legal@company.com",
        subject="Contract renewal — signature required",
        body="The vendor contract with CloudProvider Inc. expires in 10 days. The renewal agreement is ready and requires your digital signature. Please sign before Jan 25 to avoid service interruption.",
        timestamp="2024-01-15T09:00:00Z",
        true_label="action-needed",
        true_priority=7,
    ),
    Email(
        id="e8",
        sender="teammate@company.com",
        subject="PR review request — auth module refactor",
        body="Hey, I have submitted a pull request for the auth module refactor we discussed. Would appreciate your review when you get a chance this week. No rush but would be great before the sprint ends.",
        timestamp="2024-01-15T14:00:00Z",
        true_label="action-needed",
        true_priority=8,
    ),

    # --- FYI (priority 9-13) ---
    Email(
        id="e9",
        sender="newsletter@techdigest.com",
        subject="This week in AI — weekly digest",
        body="Welcome to this week's edition of Tech Digest. Top stories: OpenAI releases new model, GitHub Copilot updates, cloud pricing changes. Read more at our website.",
        timestamp="2024-01-15T06:00:00Z",
        true_label="fyi",
        true_priority=9,
    ),
    Email(
        id="e10",
        sender="office@company.com",
        subject="Office closed Monday — public holiday",
        body="Reminder that the office will be closed next Monday due to the public holiday. The building access system will be in restricted mode. Work from home if needed.",
        timestamp="2024-01-15T08:00:00Z",
        true_label="fyi",
        true_priority=10,
    ),
    Email(
        id="e11",
        sender="ceo@company.com",
        subject="Q3 results — company all-hands recording",
        body="For those who missed the all-hands meeting yesterday, the recording is now available on the intranet. Q3 results were strong. Thanks to everyone for the hard work this quarter.",
        timestamp="2024-01-15T12:00:00Z",
        true_label="fyi",
        true_priority=11,
    ),
    Email(
        id="e12",
        sender="devops@company.com",
        subject="Scheduled maintenance this Saturday 2-4am",
        body="Heads up: we will be performing scheduled maintenance on the CI/CD pipeline this Saturday between 2am and 4am UTC. Deployments will be paused during this window.",
        timestamp="2024-01-15T13:00:00Z",
        true_label="fyi",
        true_priority=12,
    ),
    Email(
        id="e13",
        sender="colleague@company.com",
        subject="Lunch moved to Thursday this week",
        body="Just a heads up — the team lunch has been moved from Wednesday to Thursday this week due to the conference room booking clash. Same time, 12:30pm, same place.",
        timestamp="2024-01-15T09:45:00Z",
        true_label="fyi",
        true_priority=13,
    ),

    # --- SPAM (priority 14-17) ---
    Email(
        id="e14",
        sender="noreply@prizes-winner.net",
        subject="You have been selected! Claim your $1000 gift card",
        body="Congratulations! You have been randomly selected to receive a $1000 Amazon gift card. Click the link below within 24 hours to claim your prize. Limited time offer!",
        timestamp="2024-01-15T05:00:00Z",
        true_label="spam",
        true_priority=14,
    ),
    Email(
        id="e15",
        sender="deals@cheapmeds-online.biz",
        subject="Best prices on prescription meds — no prescription needed",
        body="Get any medication delivered to your door. No prescription required. 80% cheaper than pharmacy prices. Order now and get free shipping on your first order.",
        timestamp="2024-01-15T04:30:00Z",
        true_label="spam",
        true_priority=15,
    ),
    Email(
        id="e16",
        sender="prince.nigeria@gmail.com",
        subject="Confidential business proposal — urgent response needed",
        body="Dear friend, I am a prince seeking to transfer $45 million USD out of my country. I need a trusted partner. You will receive 30% commission. Please respond with your bank details.",
        timestamp="2024-01-15T03:00:00Z",
        true_label="spam",
        true_priority=16,
    ),
    Email(
        id="e17",
        sender="seo@rankfast-guaranteed.com",
        subject="Get your website to #1 on Google GUARANTEED",
        body="We guarantee first page Google rankings within 30 days or your money back. Our proprietary SEO technique has worked for 10000 businesses. Free audit today — click here.",
        timestamp="2024-01-15T02:00:00Z",
        true_label="spam",
        true_priority=17,
    ),

    # --- COMPLAINT EMAIL for Task 3 (hard) ---
    Email(
        id="e18",
        sender="angry.client@corporation.com",
        subject="Extremely disappointed — 3 unresolved issues",
        body="""Dear Support Team,

I am writing to express my deep frustration with your service. I have been a paying customer for 3 years and have never experienced such poor quality.

First, your billing system charged me twice for last month's invoice. I reported this 2 weeks ago and still have not received a refund.

Second, your mobile app has been crashing every time I try to export reports. This has been broken for over a month and is blocking my entire workflow.

Third, your support team promised a callback within 24 hours last Tuesday. It has now been 6 days and nobody has contacted me.

I expect all three of these issues to be addressed in your response. If I do not receive a satisfactory reply within 48 hours, I will be cancelling my subscription and disputing the charges with my bank.

Regards,
Michael Chen
Enterprise Plan Customer""",
        timestamp="2024-01-15T08:00:00Z",
        true_label="urgent",
        true_priority=1,
    ),

    # --- Extra emails for variety ---
    Email(
        id="e19",
        sender="marketing@company.com",
        subject="Please review the new campaign copy",
        body="Hi, the Q1 marketing campaign copy is ready for your review. We need sign-off before we send to the designer. Please take a look at the attached draft and let us know your thoughts by Wednesday.",
        timestamp="2024-01-15T15:00:00Z",
        true_label="action-needed",
        true_priority=9,
    ),
    Email(
        id="e20",
        sender="events@company.com",
        subject="Company picnic — save the date",
        body="Mark your calendars! The annual company picnic is on March 15th at Riverside Park. More details to follow. Partners and families are welcome. Hope to see everyone there.",
        timestamp="2024-01-15T16:00:00Z",
        true_label="fyi",
        true_priority=20,
    ),
]


# ── Helper: return emails for each task ─────────────────────────────

def get_task_emails(task: str) -> List[Email]:
    """Return the right subset of emails per task."""

    if task == "label":
        # 8 emails — 2 of each label, good mix
        ids = ["e1", "e9", "e5", "e14", "e2", "e10", "e6", "e15"]
        return [e for e in ALL_EMAILS if e.id in ids]

    elif task == "prioritize":
        # 10 emails — mix of all labels, agent must rank them
        ids = ["e3", "e9", "e5", "e14", "e1", "e11", "e7", "e16", "e4", "e13"]
        return [e for e in ALL_EMAILS if e.id in ids]

    elif task == "reply":
        # single complaint email with 3 embedded issues
        return [e for e in ALL_EMAILS if e.id == "e18"]

    else:
        raise ValueError(f"Unknown task: {task}")


def get_ground_truth_labels(task: str) -> dict:
    """Return {email_id: true_label} for the label task."""
    emails = get_task_emails(task)
    return {e.id: e.true_label for e in emails}


def get_ground_truth_ranking(task: str) -> List[str]:
    """Return email ids sorted by true_priority ascending (1 = most urgent)."""
    emails = get_task_emails(task)
    sorted_emails = sorted(emails, key=lambda e: e.true_priority)
    return [e.id for e in sorted_emails]


# ── Issues embedded in the complaint email (for grader) ─────────────

COMPLAINT_ISSUES = [
    "double billing / charged twice / duplicate charge / refund",
    "mobile app crashing / export reports broken / app bug",
    "no callback / support never called / missed callback / 6 days no contact",
]