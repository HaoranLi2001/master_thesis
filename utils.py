import torch

def forward_KLD(
        logit_s,
        logit_t,
        targets,
        padding_id,
        temp = 1.0,
        reduction='sum'
):
    logit_s = logit_s / temp
    logit_t = logit_t / temp

    log_teacher_logit = torch.log_softmax(logit_t,-1)
    log_student_logit = torch.log_softmax(logit_s,-1)
    teacher_logit = torch.softmax(logit_t,-1)

    kl = teacher_logit * (log_teacher_logit - log_student_logit)

    # size: [batch, seq, vocab], sum over vocab
    kl = kl.sum(-1)

    if reduction == 'sum':
        padding = targets.eq(padding_id)
        kl = kl.masked_fill_(padding, 0.0)
        kl = kl.sum()

    return kl