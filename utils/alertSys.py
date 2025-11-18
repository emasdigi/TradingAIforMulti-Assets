import requests
import json


# gcloud secrets create slack-channel-urls --data-file=slackChannelUrl.json
class CentralizedSlackAlert:
    """
    Description: A class to prepare and send slack channel alert
    """

    def __init__(
        self,
        slackChannelNameUrlDict=None,
        slackIdDict=None,
        env="development",
        cloud="AWS",
    ):
        """initiate class and import a few configration parameters"""
        self.severityColorCodeDict = {
            "urgent": "#FF0000",
            "warn": "#FFBF00",
            "info": "#008000",
        }
        self.env = env
        self.cloud = cloud
        self.slackChannelNameUrlDict = slackChannelNameUrlDict
        self.slackIdDict = slackIdDict

    def slack_url(self, slackChannelName):
        if slackChannelName not in self.slackChannelNameUrlDict.keys():
            raise "please add slack group name and slack url pair into GCP/ AWS secrect manager a and try again"
        return self.slackChannelNameUrlDict[slackChannelName]

    def slack_id(self, emailAddress):
        if self.slackIdDict == None:
            with open("slackUserId.json") as f:
                self.slackIdDict = json.load(f)
        if emailAddress not in self.slackIdDict.keys():
            # raise "please add email address and slack id pair into google sheet: https://docs.google.com/spreadsheets/d/1Q5TEcEI39yv3bI_gMXynZrQLnuxpm4w4QvTKQ5Zz0JI/edit#gid=0"
            raise "please add email address and slack id pair into GCP/ AWS secrect manager and try again"
        return self.slackIdDict[emailAddress]

    def slack_alert_msg_update(self, title, value, short=False):
        short = len(str(value)) <= 20
        self.slackMsg["attachments"][0]["fields"].append(
            {"title": title, "value": str(value), "short": short}
        )

    def send_slack_msg(
        self,
        slackChannelName,
        title,
        assetClass,
        attentionEmailList,
        otherMsgs={},
        severity="info",
    ):
        """Send slack message"""
        if severity not in self.severityColorCodeDict.keys():
            severity = "info"
        severityColorCode = self.severityColorCodeDict[severity]
        self.slackUrl = self.slack_url(slackChannelName)
        self.slackMsg = {
            "text": ":pluang_new: {}".format(title),
            "attachments": [{"fields": [], "color": severityColorCode}],
        }
        self.slack_alert_msg_update("Asset Class", assetClass)
        for key in otherMsgs.keys():
            self.slack_alert_msg_update(key, otherMsgs[key])
        self.slack_alert_msg_update(
            title="Attention",
            value=", ".join(
                [
                    "<@{}>".format(self.slack_id(emailAddress))
                    for emailAddress in attentionEmailList
                ]
            ),
            short=True,
        )
        self.slack_alert_msg_update(
            title="Environment",
            value="{} - {}".format(self.cloud, self.env),
            short=True,
        )
        requests.post(
            self.slackUrl,
            headers={"Content-Type": "application/json"},
            json=self.slackMsg,
        )
