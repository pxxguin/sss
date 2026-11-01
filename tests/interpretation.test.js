import test from 'node:test';import assert from 'node:assert/strict';
import {direction,attributionText,deltaText,points} from '../web/interpretation.js';
test('positive and negative contributions explain model score, not sensor direction',()=>{assert.equal(direction(.03),'up');assert.equal(direction(-.03),'down');assert.match(attributionText('맥박수',.03),/3.00%p 높이는 기여/);assert.match(attributionText('맥박수',-.03),/3.00%p 낮추는 기여/);assert.match(attributionText('맥박수',0),/거의 0/);});
test('change is current minus original, including negative contributions',()=>{assert.match(deltaText(-.03,-.08),/5.00%p 증가/);assert.match(deltaText(.01,.08),/7.00%p 감소/);assert.match(deltaText(.01,.01),/거의 없/);assert.equal(points(-.000000001),'0.00%p');});
