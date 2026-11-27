/*
 * BetterGI Rust 脚本兼容层。
 *
 * 职责：把 C# 原版脚本使用的宿主 API 名，映射到 `__host` 提供的三个原语。
 * 这一层只做名字折叠与参数转换，不实现业务逻辑；业务能力全在 Rust HostKernel。
 *
 * C# ClearScript 开了 UseCaseInsensitiveMemberBinding，脚本里
 * `file.readTextSync` 与 `file.ReadTextSync`、`dispatcher.addTimer` 与
 * `dispatcher.AddTrigger` 混用，因此这里所有对象都必须大小写折叠。
 */
(function () {
  "use strict";

  const HOOK = globalThis.__host;
  if (!HOOK || typeof HOOK.sync !== "function" || typeof HOOK.async !== "function") {
    throw new Error("宿主未注入 __host 原语");
  }

  /** 宿主错误以纯数据返回，这里统一解包成 JS Error。 */
  function unwrap(result) {
    if (result !== null && typeof result === "object" && typeof result.$hostError === "string") {
      const e = new Error(result.message || result.$hostError);
      e.hostKind = result.$hostError;
      throw e;
    }
    return result;
  }

  /** 把方法名首字母小写，对齐 Rust 侧的 canonical_method。 */
  function canonical(name) {
    if (typeof name !== "string" || name.length === 0) return name;
    return name[0].toLowerCase() + name.slice(1);
  }

  /**
   * 大小写不敏感的对象代理：读取任意属性时先尝试规范名，
   * 再尝试原样；两者都不存在返回 undefined（而不是抛错，保持 JS 常规语义）。
   */
  function ci(target) {
    return new Proxy(target, {
      get(t, key) {
        if (typeof key !== "string") return Reflect.get(t, key);
        if (key in t) return Reflect.get(t, key);
        const c = canonical(key);
        if (c !== key && c in t) return Reflect.get(t, c);
        // 找出仅大小写不同的键（C# 侧方法名既有 PascalCase 也有 camelCase）
        for (const k of Object.keys(t)) {
          if (k.toLowerCase() === key.toLowerCase()) return Reflect.get(t, k);
        }
        return undefined;
      },
      has(t, key) {
        if (typeof key !== "string") return Reflect.has(t, key);
        return (
          Reflect.has(t, key) ||
          Object.keys(t).some((k) => k.toLowerCase() === key.toLowerCase())
        );
      },
    });
  }

  const sync = (method, args) => unwrap(HOOK.sync(method, args || []));
  const asyncCall = (method, args) => {
    const p = HOOK.async(method, args || []);
    return Promise.resolve(p).then(unwrap);
  };
  const release = (value) => {
    if (value && typeof value === "object" && typeof value.$handle === "number") {
      HOOK.release(value.$handle);
    }
  };

  // ---- 全局函数（大小写折叠靠 globals 上同时挂两种写法） ----
  const globals = {
    keyDown: (vk) => sync("keyDown", [vk]),
    keyUp: (vk) => sync("keyUp", [vk]),
    keyPress: (vk) => sync("keyPress", [vk]),
    click: (x, y) => sync("click", [x, y]),
    leftButtonClick: () => sync("leftButtonClick", []),
    leftButtonDown: () => sync("leftButtonDown", []),
    leftButtonUp: () => sync("leftButtonUp", []),
    rightButtonClick: () => sync("rightButtonClick", []),
    rightButtonDown: () => sync("rightButtonDown", []),
    rightButtonUp: () => sync("rightButtonUp", []),
    middleButtonClick: () => sync("middleButtonClick", []),
    verticalScroll: (delta) => sync("verticalScroll", [delta]),
    moveMouseBy: (dx, dy) => sync("moveMouseBy", [dx, dy]),
    moveMouseTo: (x, y) => sync("moveMouseTo", [x, y]),
    captureGameRegion: () => sync("captureGameRegion", []),
    getAvatars: () => sync("getAvatars", []),
    setGameMetrics: (metrics) => sync("setGameMetrics", [metrics]),
    getGameMetrics: () => sync("getGameMetrics", []),
    inputText: (text) => sync("inputText", [text]),
    sleep: (ms) => asyncCall("sleep", [ms]),
  };
  for (const [name, fn] of Object.entries(globals)) {
    globalThis[name] = fn;
  }

  // ---- log ----
  const countLog = () => {
    if (HOOK.__countLog) HOOK.__countLog();
  };
  const log = ci({
    info: (msg) => { countLog(); return sync("log.info", [String(msg)]); },
    warn: (msg) => { countLog(); return sync("log.warn", [String(msg)]); },
    error: (msg) => { countLog(); return sync("log.error", [String(msg)]); },
    debug: (msg) => { countLog(); return sync("log.debug", [String(msg)]); },
  });

  // ---- file ----
  const file = ci({
    readText: (path) => asyncCall("file.readText", [path]),
    writeText: (path, text) => asyncCall("file.writeText", [path, text]),
    readTextSync: (path) => sync("file.readTextSync", [path]),
    readTextAsync: (path) => asyncCall("file.readText", [path]),
    writeTextSync: (path, text) => sync("file.writeTextSync", [path, text]),
    readPathSync: (path) => sync("file.readPathSync", [path]),
    readImageMatSync: (path) => sync("file.readImageMatSync", [path]),
    readImageMatWithResizeSync: (path, w, h) =>
      sync("file.readImageMatWithResizeSync", [path, w, h]),
    writeImageSync: (path, image) => sync("file.writeImageSync", [path, image]),
    createDirectory: (path) => sync("file.createDirectory", [path]),
    isFolder: (path) => sync("file.isFolder", [path]),
    isFile: (path) => sync("file.isFile", [path]),
    isExists: (path) => sync("file.isExists", [path]),
    renamePathSync: (from, to) => sync("file.renamePathSync", [from, to]),
    deletePathSync: (path) => sync("file.deletePathSync", [path]),
  });

  // ---- genshin：只映射真实脚本用到的成员（见 API 盘点） ----
  const genshin = ci({
    returnMainUi: () => asyncCall("genshin.returnMainUi", []),
    tp: (...args) => asyncCall("genshin.tp", args),
    tpToStatueOfTheSeven: () => asyncCall("genshin.tpToStatueOfTheSeven", []),
    switchParty: (name) => asyncCall("genshin.switchParty", [name]),
    switchCharacter: (index) => asyncCall("genshin.switchCharacter", [index]),
    relogin: () => asyncCall("genshin.relogin", []),
    wonderlandCycle: () => asyncCall("genshin.wonderlandCycle", []),
    getPositionFromMap: () => sync("genshin.getPositionFromMap", []),
    getPositionFromMapWithMatchingMethod: (method) =>
      sync("genshin.getPositionFromMapWithMatchingMethod", [method]),
    getPositionFromBigMap: () => sync("genshin.getPositionFromBigMap", []),
    getCameraOrientation: () => sync("genshin.getCameraOrientation", []),
    getBigMapZoomLevel: () => sync("genshin.getBigMapZoomLevel", []),
  });

  // ---- dispatcher ----
  const dispatcher = ci({
    addTimer: (...args) => sync("dispatcher.addTimer", args),
    addTrigger: (name, options) => sync("dispatcher.addTrigger", [name, options]),
    clearAllTriggers: () => sync("dispatcher.clearAllTriggers", []),
    runTask: (...args) => asyncCall("dispatcher.runTask", args),
    runCombatScript: (...args) => asyncCall("dispatcher.runCombatScript", args),
    getLinkedCancellationToken: () => sync("dispatcher.getLinkedCancellationToken", []),
  });

  // ---- 其余对象 ----
  const notification = ci({
    send: (title, message) => sync("notification.send", [title, message]),
    error: (title, message) => sync("notification.error", [title, message]),
  });

  const pathingScript = ci({
    run: (json) => asyncCall("pathingScript.run", [json]),
    runFile: (path) => asyncCall("pathingScript.runFile", [path]),
    runFileFromUser: (path) => asyncCall("pathingScript.runFileFromUser", [path]),
  });

  const keyMouseScript = ci({
    run: (json) => asyncCall("keyMouseScript.run", [json]),
    runFile: (path) => asyncCall("keyMouseScript.runFile", [path]),
  });

  /** 资源对象：截图区域 / 识别对象 / Mat。全部带 dispose。 */
  function wrapHandle(value) {
    if (!value || typeof value !== "object" || typeof value.$handle !== "number") {
      return value;
    }
    const methods = ["find", "findMulti", "click", "isExist", "isEmpty", "ocr", "Ocr", "TemplateMatch"];
    const base = { $handle: value.$handle, $kind: value.$kind };
    base.dispose = () => release(value);
    base.text = value.text === undefined ? undefined : value.text;
    for (const m of methods) {
      base[m] = (...args) => sync("region." + canonical(m), [{ $handle: value.$handle }, ...args]);
    }
    return ci(base);
  }

  const recognition = ci({
    TemplateMatch: (source, template, options) =>
      wrapHandle(sync("RecognitionObject.TemplateMatch", [source, template, options])),
    ocr: (source, options) =>
      wrapHandle(sync("RecognitionObject.ocr", [source, options])),
  });

  // RecognitionObject 是 new 出来的类；这里提供一个可构造的包装。
  function RecognitionObject(options) {
    return recognition;
  }

  // 挂到全局（同样大小写折叠：脚本用 ocr/Ocr 都命中）
  const exported = {
    log,
    file,
    genshin,
    dispatcher,
    notification,
    pathingScript,
    keyMouseScript,
    RecognitionObject,
    dispose: release,
    settings: globalThis.__settings || {},
  };
  for (const [name, value] of Object.entries(exported)) {
    globalThis[name] = value;
  }
  // `settings` 注入值在 shim 执行前由 Rust 写到 __settings；这里覆盖掉占位。
  if (globalThis.__settings !== undefined) {
    globalThis.settings = globalThis.__settings;
  }

  // 未映射 API 的兜底：访问已知宿主对象上不存在的成员时，返回一个会抛错的函数，
  // 而不是 undefined —— 避免脚本静默继续、最后报出无关的 TypeError。
  function strictProxy(target, objectName) {
    return new Proxy(target, {
      get(t, key) {
        const v = ci(t)[key];
        if (v !== undefined) return v;
        if (typeof key === "string") {
          return () => {
            throw new Error("宿主方法 " + objectName + "." + key + " 未实现");
          };
        }
        return v;
      },
    });
  }
  for (const name of ["genshin", "dispatcher", "file", "notification", "pathingScript", "keyMouseScript", "log"]) {
    globalThis[name] = strictProxy(exported[name], name);
  }

  // 清理注入用的临时全局，避免脚本看到内部实现细节
  delete globalThis.__host;
  delete globalThis.__settings;
})();
