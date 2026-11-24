document.getElementById("save").addEventListener("click", async () => {
  const serverUrl = document.getElementById("serverUrl").value;

  await chrome.storage.sync.set({ serverUrl });

  alert("Saved");
});

(async () => {
  const data = await chrome.storage.sync.get(["serverUrl"]);

  if (data.serverUrl) {
    document.getElementById("serverUrl").value = data.serverUrl;
  }
})();