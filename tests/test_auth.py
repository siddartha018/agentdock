import asyncio

import pytest
from langgraph_sdk import Auth

from agentdock.auth import own_threads, resolve


def test_api_key_header_and_bearer():
    assert resolve({b"x-api-key": b"dk_dev_e100"})["identity"] == "E100"
    assert resolve({"authorization": "Bearer dk_dev_m200"})["role"] == "manager"


@pytest.mark.parametrize("headers", [{}, {b"x-api-key": b"nope"}, {"authorization": "Basic abc"}])
def test_bad_or_missing_key_is_401(headers):
    with pytest.raises(Auth.exceptions.HTTPException) as e:
        resolve(headers)
    assert e.value.status_code == 401


def test_service_can_delegate_and_is_recorded():
    user = resolve({b"x-api-key": b"dk_dev_svc-concierge", b"x-on-behalf-of": b"E101"})
    assert (user["identity"], user["role"], user["via"]) == ("E101", "employee", "svc-concierge")


def test_people_cannot_delegate():
    with pytest.raises(Auth.exceptions.HTTPException) as e:
        resolve({b"x-api-key": b"dk_dev_e100", b"x-on-behalf-of": b"H300"})
    assert e.value.status_code == 403


@pytest.mark.parametrize("target", ["svc-concierge", "nobody"])
def test_cannot_delegate_to_services_or_unknowns(target):
    with pytest.raises(Auth.exceptions.HTTPException):
        resolve({b"x-api-key": b"dk_dev_svc-concierge", b"x-on-behalf-of": target.encode()})


class _Ctx:
    def __init__(self, action):
        self.action = action
        self.user = type("U", (), {"identity": "E100"})()


def test_owner_stamped_into_the_servers_own_metadata_dict():
    # Regression: an empty {} is falsy; replacing it dropped the owner and hid
    # stateless MCP runs from their own caller.
    metadata = {}
    value = {"metadata": metadata}
    filters = asyncio.run(own_threads(_Ctx("create_run"), value))
    assert metadata == {"owner": "E100"} and value["metadata"] is metadata
    assert filters == {"owner": "E100"}
